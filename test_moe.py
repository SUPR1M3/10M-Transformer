import jax
import jax.numpy as jnp
from flax import nnx
from config import make_config
from data_gen import sample_batch
from transformer import MoE
from sweep import train_run, n_params


def moe_reference(moe, x):
    B, T, D = x.shape
    xf = x.reshape(-1, D)
    probs = jax.nn.softmax(moe.router(xf), axis=-1)
    gate, expert = jax.lax.top_k(probs, moe.k)
    if moe.k > 1:
        gate = gate / gate.sum(-1, keepdims=True)
    h = jax.nn.gelu(jnp.einsum("md,edf->emf", xf, moe.w_up[...]))
    y_all = jnp.einsum("emf,efd->emd", h, moe.w_down[...])#(E, M, D)
    m = jnp.arange(xf.shape[0])
    out = sum(y_all[expert[:, j], m] * gate[:, j, None] for j in range(moe.k))
    return out.reshape(B, T, D)


def test_matches_reference():
    for k in [1, 2]:
        cfg = make_config(2, 64, max_digits=5, arch="moe", n_experts=8, top_k=k)
        moe = MoE(cfg, nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(1), (4, 17, 64))
        out, aux = moe(x)
        err = float(jnp.abs(out - moe_reference(moe, x)).max())
        print(f"k={k}  shape={out.shape}  max err={err:.2e}  aux={float(aux):.3f}")
        assert out.shape == x.shape
        assert err < 1e-4


def test_router_gets_gradients():
    cfg = make_config(2, 64, max_digits=5, arch="moe", n_experts=8, top_k=1)
    moe = MoE(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (4, 17, 64))
    grads = nnx.grad(lambda m: m(x)[0].sum())(moe)
    g = float(jnp.abs(grads.router.kernel[...]).sum())
    print(f"router grad magnitude = {g:.3e}")
    assert g > 0


def test_param_counts():
    from transformer import Transformer
    L, D, E, F = 2, 64, 8, 256
    dense_cfg = make_config(2, 64, max_digits=5, arch="dense")
    moe_cfg = make_config(2, 64, max_digits=5, arch="moe", n_experts=8, top_k=1)
    d_act, d_tot, _ = n_params(Transformer(dense_cfg, nnx.Rngs(0)), dense_cfg)
    m_act, m_tot, _ = n_params(Transformer(moe_cfg, nnx.Rngs(0)), moe_cfg)
    router = L * D * E
    dense_mlp_biases = L * (F + D)
    print(f"dense: active={d_act:,} total={d_tot:,}")
    print(f"moe: active={m_act:,} total={m_tot:,}")
    assert d_act == d_tot
    assert m_act > d_act
    assert m_act - d_act == router - dense_mlp_biases
    assert m_tot > 5 * d_tot


def test_smoke_train(arch):
    cfg = make_config(2, 48, max_digits=10, arch=arch, n_experts=8, top_k=1)
    eval_data = sample_batch(jax.random.key(12345), 2_000, 10)
    r = train_run(cfg, n_tokens=300_000, eval_data=eval_data, eval_every=20)
    print(f"  {arch}: N={r['N']:,}  N_total={r['N_total']:,}  "
          f"first={r['curve'][0][1]:.3f}  last={r['curve'][-1][1]:.3f}  final={r['loss']:.3f}")
    assert r["curve"][-1][1] < r["curve"][0][1]       # loss went down


if __name__ == "__main__":
    print("1. MoE matches reference");  test_matches_reference()
    print("2. router gets gradients");  test_router_gets_gradients()
    print("3. parameter counts");       test_param_counts()
    print("4. smoke train (dense)");    test_smoke_train("dense")
    print("5. smoke train (moe)");      test_smoke_train("moe")
    print("\nall tests passed")