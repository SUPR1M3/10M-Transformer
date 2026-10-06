import jax.numpy as jnp


def make_config(L, D, max_digits=5, B=64, lr=1e-3, head_dim=16, seed=0, ff_mult=4):
    assert D % head_dim == 0, f"D={D} not divisible by head_dim={head_dim}"
    return {
        "B": B, "D": D, "F": ff_mult * D, "F/D": ff_mult,
        "L": L, "H": D // head_dim,
        "max_digits": max_digits, "seq_len": 3 * max_digits + 3,
        "seed": seed, "vocab_size": 13, "learning_rate": lr,
    }


def make_mask(config):
    # Loss only on answer tokens: targets at index >= 2*max_digits+1 (everything after "=")
    return (jnp.arange(config["seq_len"] - 1) >= 2 * config["max_digits"] + 1).astype(jnp.float32)


class Config:
    def __init__(self):
        self.config = make_config(L=6, D=384, max_digits=3, head_dim=64)
