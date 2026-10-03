
import sys, os
import jax, jax.numpy as jnp, equinox as eqx
from typing import Optional, Sequence

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from spectralconv1d_jax import SpectralConv1d, Conv1dResidual


class MLP(eqx.Module):
    layers: tuple

    def __init__(self, layer_sizes: Sequence[int], *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(layer_sizes) - 1)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(
                eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1],
                              key=keys[i], use_bias=True))
        self.layers = tuple(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jnp.tanh(x)
        return x




class DFNOSeparableDeepONet3d(eqx.Module):

    fc0: eqx.nn.Linear
    weight_x: jnp.ndarray
    weight_y: jnp.ndarray

    conv_x: tuple; w_x: tuple
    conv_y: tuple; w_y: tuple

    branch_head: eqx.nn.Linear

    trunk_x: MLP; trunk_y: MLP; trunk_z: MLP

    output_bias: jnp.ndarray

    coords_x_default: jnp.ndarray
    coords_y_default: jnp.ndarray
    coords_z_default: jnp.ndarray

    R: int = eqx.field(static=True)
    nx: int = eqx.field(static=True); ny: int = eqx.field(static=True); nz: int = eqx.field(static=True)
    p: int = eqx.field(static=True); width: int = eqx.field(static=True)
    P: int = eqx.field(static=True); modes: int = eqx.field(static=True)
    num_fourier_layers: int = eqx.field(static=True)

    def __init__(self, trunk_layers: Sequence[int], R: int = 50,
                 nx: int = 64, ny: int = 64, nz: int = 48,
                 width: int = 64, P: int = 25, modes: int = 7,
                 num_fourier_layers: int = 6, *, key: jax.random.PRNGKey):
        super().__init__()
        self.R, self.nx, self.ny, self.nz = R, nx, ny, nz
        self.p = trunk_layers[-1]
        self.width, self.P, self.modes = width, P, modes
        self.num_fourier_layers = num_fourier_layers

        s = max(nx, ny)
        k_fc0, k_bh, k_tx, k_ty, k_tz, k_conv = jax.random.split(key, 6)
        N = num_fourier_layers

        self.fc0 = eqx.nn.Linear(1, width, key=k_fc0, use_bias=True)

        self.weight_x = jax.random.normal(k_conv, (P, s))
        self.weight_y = jax.random.normal(k_conv, (P, s))

        rfft_len = s // 2 + 1
        M = min(modes, rfft_len - 2)
        keys_c = jax.random.split(k_conv, 4 * N)
        ci = 0

        cx_list = []; wx_list = []; cy_list = []; wy_list = []
        for i in range(N):
            cx_list.append(SpectralConv1d(width, width, M, s, keys_c[ci])); ci += 1
        for i in range(N):
            wx_list.append(Conv1dResidual(width, keys_c[ci])); ci += 1
        for i in range(N):
            cy_list.append(SpectralConv1d(width, width, M, s, keys_c[ci])); ci += 1
        for i in range(N):
            wy_list.append(Conv1dResidual(width, keys_c[ci])); ci += 1
        self.conv_x = tuple(cx_list); self.w_x = tuple(wx_list)
        self.conv_y = tuple(cy_list); self.w_y = tuple(wy_list)

        self.branch_head = eqx.nn.Linear(width, self.p, key=k_bh, use_bias=True)

        hidden = list(trunk_layers[:-1])
        self.trunk_x = MLP([1] + hidden + [trunk_layers[-1] * R], key=k_tx)
        self.trunk_y = MLP([1] + hidden + [trunk_layers[-1] * R], key=k_ty)
        self.trunk_z = MLP([1] + hidden + [trunk_layers[-1] * R], key=k_tz)

        self.output_bias = jnp.zeros(())
        self.coords_x_default = jnp.linspace(0, 1, nx)
        self.coords_y_default = jnp.linspace(0, 1, ny)
        self.coords_z_default = jnp.linspace(0, 0.75, nz)

    def _dfno_feature_extract(self, x: jnp.ndarray) -> jnp.ndarray:
        B = x.shape[0]
        s_max = max(self.nx, self.ny)
        width, P = self.width, self.P

        B, s1, s2, _c = x.shape
        x = x.reshape(-1, 1)
        x = jax.vmap(self.fc0)(x)
        x = x.reshape(B, s1, s2, self.width)
        v0 = jnp.transpose(x, (0, 3, 1, 2))

        Vx = jnp.einsum('bchw,ph->bpcw', v0, self.weight_x[:, :s1])
        Vy = jnp.einsum('bchw,pw->bpch', v0, self.weight_y[:, :s2])

        Vx = Vx.reshape(-1, width, s1)
        Vy = Vy.reshape(-1, width, s2)

        v = jax.vmap
        N = self.num_fourier_layers
        for i in range(N):
            x1 = v(self.conv_x[i])(Vx); x2 = v(self.w_x[i])(Vx)
            Vx = x1 + x2 if i == N - 1 else jax.nn.gelu(x1 + x2)
        Vx_final = Vx.reshape(B, P, width, s1)

        for i in range(N):
            y1 = v(self.conv_y[i])(Vy); y2 = v(self.w_y[i])(Vy)
            Vy = y1 + y2 if i == N - 1 else jax.nn.gelu(y1 + y2)
        Vy_final = Vy.reshape(B, P, width, s2)

        vT = jnp.einsum('bpcw,bpch->bchw', Vx_final, Vy_final)
        return vT

    def __call__(self, source: jnp.ndarray,
                 coords_x: Optional[jnp.ndarray] = None,
                 coords_y: Optional[jnp.ndarray] = None,
                 coords_z: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        B = source.shape[0]
        nx, ny, nz = self.nx, self.ny, self.nz
        R, p = self.R, self.p

        if coords_x is None: coords_x = self.coords_x_default
        if coords_y is None: coords_y = self.coords_y_default
        if coords_z is None: coords_z = self.coords_z_default

        a = source[..., None]
        features = self._dfno_feature_extract(a)
        Bf = features.transpose((0, 2, 3, 1))
        Bf = jax.vmap(jax.vmap(jax.vmap(self.branch_head)))(Bf)

        phi = jax.vmap(self.trunk_x)(coords_x[:, None]).reshape(nx, R, p, 1)
        psi = jax.vmap(self.trunk_y)(coords_y[:, None]).reshape(ny, R, p, 1)
        zeta = jax.vmap(self.trunk_z)(coords_z[:, None]).reshape(nz, R, p, 1)
        trunk_3d = jnp.einsum('xRpo,yRpo,zRpo->xyzpo', phi, psi, zeta).squeeze(-1)

        u = jnp.einsum('bijp,ijkp->bijk', Bf, trunk_3d)
        return u + self.output_bias




def count_params(model: eqx.Module) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(leaf.size for leaf in leaves)


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    trunk_layers = [50, 50, 50, 50, 50, 50, 50]

    for name, cls in [('D-FNO-BPSS', DFNOSeparableDeepONet3d),
                       ('D-FNO (global pool)', DFNODeepONet3d)]:
        model = cls(trunk_layers=trunk_layers, R=50, nx=64, ny=64, nz=48,
                    width=64, P=25, modes=7, num_fourier_layers=6, key=key)
        print(f'{name}: params={count_params(model):,}')
        src = jnp.ones((2, 64, 64))
        out = model(src)
        print(f'  {src.shape} -> {out.shape}')
