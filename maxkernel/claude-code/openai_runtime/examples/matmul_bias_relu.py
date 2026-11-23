"""Small reproducible JAX input for testing the complete pipeline on a TPU.

This is an example workload, not a performance claim or the user's GQA decode.
"""

import jax
import jax.numpy as jnp


def computation(a, b, bias):
  return jax.nn.relu(jnp.matmul(a, b) + bias)


def get_inputs():
  keys = jax.random.split(jax.random.key(0), 3)
  return [
    jax.random.normal(keys[0], (512, 512), dtype=jnp.bfloat16),
    jax.random.normal(keys[1], (512, 512), dtype=jnp.bfloat16),
    jax.random.normal(keys[2], (512,), dtype=jnp.bfloat16),
  ]
