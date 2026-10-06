import jax
import jax.numpy as jnp
import numpy as np
from config import Config
from tokenizer import Tokenizer 
from functools import partial



def gen_data(config, tokenizer, test_split = 0.05):
    n= config["max_digits"]
    data = []
    lim = 10**n
    for a in range(lim):
        for b in range(lim):
            c = a+b
            a_st = str(a).rjust(n)
            b_st = str(b).rjust(n)
            c_st = str(c).rjust(n+1)
            data.append(a_st + "+" + b_st + "=" + c_st)
    encoded_data = np.array([tokenizer.encode(s) for s in data], dtype=np.int32)
    rng = np.random.default_rng(config["seed"])
    rng.shuffle(encoded_data)
    test_ct = int(len(data) * test_split)
    return encoded_data[test_ct:], encoded_data[:test_ct]


@partial(jax.jit, static_argnums=(1, 2))
def sample_batch(key, B, n):
    ka, kb = jax.random.split(key)
    da = jax.random.randint(ka, (B, n), 0, 10)       # digits, most significant first
    db = jax.random.randint(kb, (B, n), 0, 10)

    def add_column(carry, cols):                      # one column, right to left
        x, y = cols
        s = x + y + carry
        return s // 10, s % 10

    carry, dc = jax.lax.scan(add_column, jnp.zeros(B, jnp.int32),
                             (da.T[::-1], db.T[::-1]))           # least significant first
    dc = jnp.concatenate([carry[None], dc[::-1]], axis=0).T      # (B, n+1), most significant first

    def pad(d):                                                  # leading zeros → space
        seen = (jnp.cumsum(d, axis=1) > 0).at[:, -1].set(True)
        return jnp.where(seen, d, 12)

    plus, eq = jnp.full((B, 1), 10), jnp.full((B, 1), 11)
    return jnp.concatenate([pad(da), plus, pad(db), eq, pad(dc)], axis=1)


if __name__ == "__main__":
    config = Config().config
    tokenizer = Tokenizer()
    # train, test = gen_data(config, tokenizer)
    # print(train.shape)
    # print(test.shape)
    # print(repr(tokenizer.decode(train[0])))
    # print(tokenizer.decode(tokenizer.encode("123+ 45= 168")) == "123+ 45= 168")
    n = 10
    batch = sample_batch(jax.random.key(0), 1000, n)

    for r in batch:
        s = tokenizer.decode(r)
        assert len(s) == 3 * n + 3, s                       # fixed length (33 for n=10)
        assert s[n] == "+" and s[2 * n + 1] == "=", s       # symbols in the right positions
        a, b, c = int(s[:n]), int(s[n + 1:2 * n + 1]), int(s[2 * n + 2:])
        assert a + b == c, s                                 # the arithmetic is correct

    print("sampler OK")
    for r in batch[:3]:
        print(repr(tokenizer.decode(r)))