import jax
import jax.numpy as jnp
from flax import nnx
from tokenizer import Tokenizer
from config import Config
from data_gen import gen_data
import numpy as np
from checkpoint import load_model


@nnx.jit
def forward(model, x):
    return model(x)

def generate(model, config, data):
    p = 2 * config["max_digits"] + 2                  # prompt length through "=" → 8
    seq = jnp.asarray(data).at[:, p:].set(0)    # erase the true answer
    for t in range(p, config["seq_len"]):             # fill positions 8..11
        logits = forward(model, seq[:, :-1])    # (B, 11, V)
        nxt = jnp.argmax(logits[:, t - 1], axis=-1)   # output at t-1 predicts token t
        seq = seq.at[:, t].set(nxt)
    return seq


def n_carries(a, b, n):
    carries, carry = np.zeros_like(a), np.zeros_like(a)
    for _ in range(n):
        carry = ((a % 10 + b % 10 + carry) >= 10).astype(int)
        carries += carry
        a, b = a // 10, b // 10
    return carries

def evaluate(model, config, tok, data, chunk=5000):
    p = 2 * config["max_digits"] + 2
    correct = []
    for i in range(0, len(data), chunk):
        batch = data[i:i + chunk]
        pred = np.asarray(generate(model, config, batch))
        correct.append((pred[:, p:] == batch[:, p:]).all(axis=1))
    correct = np.concatenate(correct)

    strs = [tok.decode(r) for r in data]
    a = np.array([int(s[:config["max_digits"]]) for s in strs])
    b = np.array([int(s[config["max_digits"] + 1:2 * config["max_digits"] + 1]) for s in strs])
    c = n_carries(a, b, config["max_digits"])

    print(f"exact match: {correct.mean():.4%}  ({(~correct).sum()} wrong / {len(correct)})")
    for k in range(config["max_digits"] + 1):
        sel = c == k
        print(f"  {k} carries: {correct[sel].mean():.4%}  (n={sel.sum()})")
    return correct

if __name__ == "__main__":
    config =Config().config
    tokenizer = Tokenizer()
    train, test = gen_data(config, tokenizer)
    model = load_model(config)
    correct = evaluate(model, config, tokenizer, test)

    for r in test[:5]:
        print(repr(tokenizer.decode(generate(model, config, r[None])[0])))

    wrong = test[~correct]
    for r in wrong[:10]:
        print("true:", repr(tokenizer.decode(r)),
              " pred:", repr(tokenizer.decode(generate(model, config, r[None])[0])))