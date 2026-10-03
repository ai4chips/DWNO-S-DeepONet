import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
_jax_dir = os.path.join(_this_dir, '..')
sys.path.insert(0, _jax_dir)

import jax, jax.numpy as jnp, equinox as eqx


def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


class Dense(eqx.Module):
    w: jnp.ndarray; b: jnp.ndarray
    def __init__(self, in_f, out_f, key):
        self.w = jax.random.normal(key, (out_f, in_f)) * (1.0/in_f**0.5)
        self.b = jnp.zeros(out_f)
    def __call__(self, x): return x @ self.w.T + self.b


class SpectralConv1d(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    in_ch: int = eqx.field(static=True)
    out_ch: int = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    L: int = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, modes, L, key):
        self.in_ch, self.out_ch, self.modes, self.L = in_ch, out_ch, modes, L
        scale = (2.0/(in_ch+out_ch))**0.5
        kr, ki, kb = jax.random.split(key, 3)
        self.weight = scale*(jax.random.normal(kr,(in_ch,out_ch,modes))
                              +1j*jax.random.normal(ki,(in_ch,out_ch,modes)))
        self.bias = scale*jax.random.normal(kb, (out_ch, L))

    def __call__(self, x):
        L = x.shape[-1]
        x_ft = jnp.fft.rfft(x)
        x_ft = jnp.fft.fftshift(x_ft, axes=(-1,))
        center = x_ft.shape[-1]//2
        k_neg, k_pos = self.modes//2, self.modes - self.modes//2
        slc = slice(center-k_neg, center+k_pos)

        out_ft = jnp.zeros((self.out_ch, x_ft.shape[-1]), dtype=jnp.complex64)
        out_ft = out_ft.at[:, slc].set(
            jnp.einsum('ih,ioh->oh', x_ft[:, slc], self.weight))
        out_ft = jnp.fft.ifftshift(out_ft, axes=(-1,))
        return jnp.fft.irfft(out_ft, n=L).real + self.bias


class Conv1dResidual(eqx.Module):
    conv: eqx.nn.Conv1d
    def __init__(self, width, key):
        super().__init__()
        self.conv = eqx.nn.Conv1d(width, width, 1, key=key)
    def __call__(self, x): return self.conv(x)


class DFNO2d(eqx.Module):
    fc0: Dense
    weight_x: jnp.ndarray; weight_y: jnp.ndarray
    conv_x: tuple; w_x: tuple; conv_y: tuple; w_y: tuple
    fc1: Dense; fc2: Dense
    width: int = eqx.field(static=True)
    s: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    num_fourier_layers: int = eqx.field(static=True)

    def __init__(self, width=64, s=64, P=25, modes=12,
                 num_fourier_layers=6, key=None):
        self.width, self.s, self.P = width, s, P
        self.modes = modes
        self.num_fourier_layers = num_fourier_layers
        N = num_fourier_layers
        if key is None: key = jax.random.PRNGKey(0)

        k_fc0, k_wx, k_wy, k_fc1, k_fc2, *keys_c = jax.random.split(key, 5+4*N)
        self.fc0 = Dense(1, width, k_fc0)
        self.weight_x = jax.random.normal(k_wx, (P, s))
        self.weight_y = jax.random.normal(k_wy, (P, s))

        rfft_len = s//2 + 1
        M = min(modes, rfft_len - 2)

        cx_list, wx_list = [], []
        for i in range(N):
            cx_list.append(SpectralConv1d(width, width, M, s, keys_c[i]))
            wx_list.append(Conv1dResidual(width, keys_c[N+i]))
        self.conv_x, self.w_x = tuple(cx_list), tuple(wx_list)

        cy_list, wy_list = [], []
        for i in range(N):
            cy_list.append(SpectralConv1d(width, width, M, s, keys_c[2*N+i]))
            wy_list.append(Conv1dResidual(width, keys_c[3*N+i]))
        self.conv_y, self.w_y = tuple(cy_list), tuple(wy_list)

        self.fc1 = Dense(width, 512, k_fc1)
        self.fc2 = Dense(512, 1, k_fc2)

    def __call__(self, x):
        B, N = x.shape[0], self.num_fourier_layers
        v = jax.nn.gelu(self.fc0(x))
        v0 = v.transpose(0, 3, 1, 2)

        Vx = jnp.einsum('bchw,ph->bpcw', v0, self.weight_x)
        Vy = jnp.einsum('bchw,pw->bpch', v0, self.weight_y)

        vf = jax.vmap
        Vx_r = Vx.reshape(-1, self.width, self.s)
        for i in range(N):
            x1 = vf(self.conv_x[i])(Vx_r); x2 = vf(self.w_x[i])(Vx_r)
            Vx_r = x1+x2 if i==N-1 else jax.nn.gelu(x1+x2)
        Vx = Vx_r.reshape(B, self.P, self.width, self.s)

        Vy_r = Vy.reshape(-1, self.width, self.s)
        for i in range(N):
            y1 = vf(self.conv_y[i])(Vy_r); y2 = vf(self.w_y[i])(Vy_r)
            Vy_r = y1+y2 if i==N-1 else jax.nn.gelu(y1+y2)
        Vy = Vy_r.reshape(B, self.P, self.width, self.s)

        vT = jnp.einsum('bpcw,bpch->bchw', Vx, Vy)
        vT = vT.transpose(0, 2, 3, 1)
        vT = jnp.tanh(self.fc1(vT))
        return self.fc2(vT)
