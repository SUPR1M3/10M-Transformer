import jax
from config import make_config, make_mask
from data_gen import sample_batch
from flax import nnx
from transformer import Transformer, Optimizer, training_step, jitted_loss
import csv, json, os

def n_params(model, exclude_embeddings=True):
    state = nnx.state(model, nnx.Param)
    total = sum(p.size for p in jax.tree.leaves(state))
    if exclude_embeddings:
        emb = sum(p.size for p in jax.tree.leaves(nnx.state(model.embeddings, nnx.Param)))
        emb += model.head.kernel.size
        total -= emb
    return total

def train_run(config, n_tokens, eval_data, eval_every=None, curve_eval_size=4096):
    T = config["seq_len"] - 1
    steps = n_tokens // (config["B"] * T)
    assert steps > 0, f"budget too small: {n_tokens} tokens < one batch"
    model = Transformer(config, nnx.Rngs(config["seed"]))
    optimizer = Optimizer(model, config, stepct = steps)
    mask = make_mask(config)
    key = jax.random.key(config["seed"] + 1)
    ex, ey = eval_data[:, :-1], eval_data[:, 1:]
    cx, cy = ex[:curve_eval_size], ey[:curve_eval_size]

    curve = []
    for s in range(steps):
        key, sub = jax.random.split(key)
        batch = sample_batch(sub, config["B"], config["max_digits"])
        training_step(model, optimizer, batch[:, :-1], batch[:, 1:], mask)
        if eval_every and (s + 1) % eval_every == 0:
            curve.append(((s + 1) * config["B"] * T, float(jitted_loss(model, cx, cy, mask))))

    final = float(jitted_loss(model, ex, ey, mask))
    return {"N": n_params(model), "N_total": n_params(model, False),
            "D": steps * config["B"] * T, "steps": steps, "loss": final, "curve": curve}

# eval_data = sample_batch(jax.random.key(12345), 20_000, 5)   # fixed held-out set

#dry run with 55k, 450k and 3M
# for L, D in [(2, 48), (3, 112), (5, 224)]:
#     for lr in [3e-4, 1e-3, 3e-3]:
#         cfg = make_config(L, D, lr=lr)
#         r = train_run(cfg, n_tokens=50_000_000, eval_data=eval_data, eval_every=500)
#         print(L, D, lr, r["N"], r["loss"])



RESULTS = "results/runs.csv"
FIELDS = ["arch", "max_digits", "L", "D_model", "lr", "C", "C_actual", "N", "N_total", "D", "steps", "loss", "tokens_per_s", "seed"]

def log_run(row, path=RESULTS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})

def save_pilot(result, cfg, path="results/pilot"):
    os.makedirs(path, exist_ok=True)
    name = f"n{cfg['max_digits']}_L{cfg['L']}_D{cfg['D']}_lr{cfg['learning_rate']:g}.json"
    with open(os.path.join(path, name), "w") as f:
        json.dump({**result, "lr": cfg["learning_rate"], "max_digits": cfg["max_digits"]}, f)