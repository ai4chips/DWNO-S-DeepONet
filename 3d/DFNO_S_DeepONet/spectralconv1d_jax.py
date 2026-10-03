import jax, jax.numpy as jnp, equinox as eqx


class SpectralConv1d(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    in_ch: int = eqx.field(static=True)
    out_ch: int = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    L: int = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, modes, L, key):
        self.in_ch, self.out_ch, self.modes, self.L = in_ch, out_ch, modes, L
        scale = (2.0 / (in_ch + out_ch)) ** 0.5
        kr, ki, kb = jax.random.split(key, 3)
        self.weight = scale * (jax.random.normal(kr, (in_ch, out_ch, modes))
                               + 1j * jax.random.normal(ki, (in_ch, out_ch, modes)))
        self.bias = scale * jax.random.normal(kb, (out_ch, L))

    def __call__(self, x):
        L = x.shape[-1]
        x_ft = jnp.fft.rfft(x)
        x_ft = jnp.fft.fftshift(x_ft, axes=(-1,))
        center = x_ft.shape[-1] // 2
        k_neg, k_pos = self.modes // 2, self.modes - self.modes // 2
        slc = slice(center - k_neg, center + k_pos)
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
