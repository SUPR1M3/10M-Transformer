import csv, os, time
import jax
from flax import nnx
from config import make_config
from data_gen import sample_batch
from transformer import Transformer
from sweep import train_run, log_run, n_params, RESULTS

N_DIGITS   = 10
ARCH       = os.environ.get("ARCH", "dense")
SEED       = int(os.environ.get("SEED", 0))
DRY_RUN    = os.environ.get("DRY_RUN") == "1"
MAX_TOKENS = int(float(os.environ.get("MAX_TOKENS", 4e8)))
BUDGETS    = [float(x) for x in os.environ.get("BUDGETS", "3e12,1e13,3e13,1e14,3e14,1e15").split(",")]
LADDER     = [(1, 32), (1, 48), (2, 48), (2, 64), (2, 96), (3, 112),
              (4, 128), (4, 176), (5, 224), (6, 256), (6, 384)]
MIN_STEPS  = 200

def lr_rule(N):
    return min(3e-3, max(7e-4, 3e-3 * (N / 56_000) ** -0.276))

def done_keys(path=RESULTS):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["arch"], int(r["L"]), int(r["D_model"]), float(r["C"]), int(r["seed"]))
                for r in csv.DictReader(f)}

def build_plan():
    plan = []
    for L, D in LADDER:
        cfg = make_config(L, D, max_digits=N_DIGITS, arch=ARCH, seed=SEED)
        N = n_params(Transformer(cfg, nnx.Rngs(0)), cfg)[0]
        T = cfg["seq_len"]-1
        for C in BUDGETS:
            tokens = int(C/(6 * N))
            if tokens//(cfg["B"] * T) < MIN_STEPS or tokens > MAX_TOKENS:
                continue
            plan.append((C, L, D, N, tokens))
    return sorted(plan) #Smallest budget runs first

if __name__ == "__main__":
    plan = build_plan()
    done = done_keys()
    todo = [p for p in plan if (ARCH, p[1], p[2], p[0], SEED) not in done]
    est_h = sum(tok for *_, tok in todo) / 300000 / 3600
    print(f"{ARCH} seed={SEED}: {len(plan)} runs planned, {len(todo)} to do, ~{est_h:.1f} h")
    for C, L, D, N, tok in plan:
        print(f"  C={C:.0e}  L={L} D={D:<4} N={N:>10,}  tokens={tok:>12,}  lr={lr_rule(N):.2e}")
    if DRY_RUN:
        raise SystemExit
    eval_data = sample_batch(jax.random.key(12345), 20_000, N_DIGITS)
    for C, L, D, N, tokens in todo:
        cfg = make_config(L, D, max_digits=N_DIGITS, arch=ARCH, seed=SEED, lr=lr_rule(N))
        t0 = time.time()
        r = train_run(cfg, n_tokens=tokens, eval_data=eval_data)
        dt = time.time() - t0
        log_run({"arch": ARCH, "max_digits": N_DIGITS, "L": L, "D_model": D,
                 "lr": cfg["learning_rate"], "C": C, "C_actual": 6 * r["N"] * r["D"],
                 "N": r["N"], "N_total": r["N_total"], "N_with_emb": r["N_with_emb"],
                 "D": r["D"], "steps": r["steps"], "loss": r["loss"], "aux_loss": r["aux_loss"],
                 "tokens_per_s": r["D"]/dt, "seed": SEED})
        print(f"C={C:.0e} L={L} D={D} N={r['N']:,} loss={r['loss']:.3e} "
              f"aux_loss={r['aux_loss']:.2f} {r['D'] / dt:,.0f} tok/s ({dt / 60:.1f} min)", flush=True)
