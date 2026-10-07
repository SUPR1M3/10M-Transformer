import time
import jax
from config import make_config
from data_gen import sample_batch
from sweep import train_run, save_pilot
import os
TOKENS = int(os.environ.get("TOKENS", 50_000_000))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 500))
N_DIGITS = 10
eval_data = sample_batch(jax.random.key(12345), 20_000, N_DIGITS)

#55k, 450k, 3M
for L, D in [(2, 48), (3, 112), (5, 224)]:
    for lr in [1e-3, 3e-3]:
        cfg = make_config(L, D, max_digits=N_DIGITS, lr=lr)
        t0 = time.time()
        r = train_run(cfg, n_tokens=TOKENS, eval_data=eval_data, eval_every=EVAL_EVERY)
        dt = time.time() - t0
        save_pilot(r, cfg)
        print(f"L={L} D={D} lr={lr:g}  N={r['N']:,}  final loss={r['loss']:.3e}  "
              f"{r['D'] / dt:,.0f} tokens/s  ({dt / 60:.1f} min)")