import jax.random
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class KeyGenerator:
    """Class for initializing and generating new jax random keys"""

    def __init__(self, randkey=0):
        self.randkey = init_randkey(randkey)

    @jax.jit
    def with_newkey(self):
        self.randkey = gen_new_key(self.randkey)
        return self

    def tree_flatten(self):
        children = (self.randkey,)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


def init_randkey(randkey):
    """Check that randkey is a PRNG key or create one from an int"""
    if isinstance(randkey, int):
        randkey = jax.random.key(randkey)
    else:
        msg = f"Invalid {type(randkey)=}: Must be int or PRNG Key"
        assert hasattr(randkey, "dtype"), msg
        assert jnp.issubdtype(randkey.dtype, jax.dtypes.prng_key), msg

    return randkey


@jax.jit
def gen_new_key(randkey):
    """Split PRNG key to generate a new one"""
    return jax.random.split(randkey, 1)[0]
