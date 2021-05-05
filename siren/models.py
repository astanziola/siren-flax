from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import uniform as uniform_init
from jax import lax
from jax.random import uniform
from typing import Any, Callable, Sequence, Tuple
from functools import partial
import jax

Array = Any


def siren_init(weight_std, dtype):
    def init_fun(key, shape, dtype=dtype):
        return uniform(key, shape, dtype) * 2 * weight_std - weight_std

    return init_fun


def grid_init(grid_dimension, dtype):
    def init_fun(dtype=dtype):
        coord_axis = [jnp.linspace(-1, 1, d) for d in grid_dimension]
        grid = jnp.stack(jnp.meshgrid(*coord_axis), -1)
        return jnp.asarray(grid, dtype)

    return init_fun


class Sine(nn.Module):
    w0: float = 1.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        return jnp.sin(self.w0 * inputs)


class SirenLayer(nn.Module):
    features: int = 32
    w0: float = 1.0
    c: float = 6.0
    is_first: bool = False
    use_bias: bool = True
    act: Callable = jnp.sin
    precision: Any = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        input_dim = inputs.shape[-1]

        # Linear projection with init proposed in SIREN paper
        weight_std = (
            (1 / input_dim) if self.is_first else jnp.sqrt(self.c / input_dim) / self.w0
        )

        kernel = self.param(
            "kernel", siren_init(weight_std, self.dtype), (input_dim, self.features)
        )
        kernel = jnp.asarray(kernel, self.dtype)

        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.use_bias:
            bias = self.param("bias", uniform, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias

        return Sine(self.w0, self.dtype)(y)


class Siren(nn.Module):
    hidden_dim: int = 256
    output_dim: int = 3
    num_layers: int = 5
    w0: float = 1.0
    w0_first_layer: float = 30.0
    use_bias: bool = True
    final_activation: Callable = lambda x: x  # Identity
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = jnp.asarray(inputs, self.dtype)
        input_dim = x.shape[-1]

        for layernum in range(self.num_layers - 1):
            is_first = layernum == 0

            x = SirenLayer(
                features=self.hidden_dim,
                w0=self.w0_first_layer if is_first else self.w0,
                is_first=is_first,
                use_bias=self.use_bias,
            )(x)

        # Last layer, with different activation function
        x = SirenLayer(
            features=self.output_dim,
            w0=self.w0,
            is_first=False,
            use_bias=self.use_bias,
            act=self.final_activation,
        )(x)

        return x
