import jax
import jax.numpy as jnp
from flax import nnx
from config import Config
import optax

class Embeddings(nnx.Module):
    def __init__(self, config, rngs):
        self.gen_emb = nnx.Embed(config["vocab_size"], config["D"], rngs=rngs)
        self.pos_emb = nnx.Embed(config["seq_len"], config["D"], rngs=rngs)

    def __call__(self, x):
        return self.gen_emb(x) + self.pos_emb(jnp.arange(x.shape[1]))



class AttentionBlock(nnx.Module):
    def __init__(self, config, rngs):
        self.H, self.Dh = config["H"], config["D"]//config["H"]
        self.qkv = nnx.Linear(config["D"], 3*config["D"], use_bias=False, rngs=rngs)
        self.out = nnx.Linear(config["D"], config["D"], use_bias=False, rngs=rngs)

    def __call__(self, x):
        B, S, D = x.shape
        q,k,v = jnp.split(self.qkv(x), 3, axis = -1)
        q = q.reshape(B, S, self.H, self.Dh)
        k = k.reshape(B, S, self.H, self.Dh)
        v = v.reshape(B, S, self.H, self.Dh)

        #TODO: Add RoPE here later

        scores = jnp.einsum('bqhd,bkhd->bhqk', q,k)/ jnp.sqrt(self.Dh)
        mask = jnp.tril(jnp.ones((S,S), dtype = bool))
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

        weights = jax.nn.softmax(scores, axis = -1)
        out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
        return self.out(out.reshape(B,S,D))


class MLP(nnx.Module):
    def __init__(self, config, rngs):
        self.up_proj = nnx.Linear(config["D"], config["F"], rngs=rngs)
        self.down_proj = nnx.Linear(config["F"], config["D"], rngs=rngs)
    
    def __call__(self, x):
        return self.down_proj(jax.nn.gelu(self.up_proj(x))), 0.0

class TransformerBlock(nnx.Module):
    def __init__(self, config, rngs):
        self.ln1 = nnx.LayerNorm(config["D"], rngs=rngs)
        self.attention = AttentionBlock(config, rngs)
        self.ln2 = nnx.LayerNorm(config["D"], rngs=rngs)
        self.mlp = MoE(config, rngs) if config.get("arch", "dense") == "moe" else MLP(config, rngs)

    def __call__(self, x):
        x = x + self.attention(self.ln1(x))
        h, aux_loss = self.mlp(self.ln2(x))
        return x + h, aux_loss

class Transformer(nnx.Module):
    def __init__(self, config, rngs):
        self.embeddings = Embeddings(config, rngs)
        self.blocks = nnx.List(TransformerBlock(config, rngs) for _ in range(config["L"]))
        self.final_layernorm = nnx.LayerNorm(config["D"], rngs = rngs)
        self.head = nnx.Linear(config["D"], config["vocab_size"], use_bias = False, rngs=rngs)

    def __call__(self, x):
        activations = self.embeddings(x)
        total_aux_loss = 0.0
        for block in self.blocks:
            activations, aux_loss = block(activations)
            total_aux_loss+= aux_loss
        activations = self.final_layernorm(activations)
        return self.head(activations), total_aux_loss / len(self.blocks)


def ce_loss(model, In, Out, mask):
    logits, aux_loss = model(In)
    CE_loss = optax.softmax_cross_entropy_with_integer_labels(logits, Out)
    return (CE_loss * mask).sum() / (In.shape[0] * mask.sum()), aux_loss


def calc_total_loss(model, In, Out, mask, aux_factor):
    CE_loss, aux_loss = ce_loss(model, In, Out, mask)
    return CE_loss + aux_factor * aux_loss

@nnx.jit
def jitted_loss(model, In, Out, mask):
    return ce_loss(model, In, Out, mask)

def Optimizer(model, config, stepct):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, 
        peak_value = config["learning_rate"], 
        warmup_steps = int(0.05*stepct), 
        decay_steps= stepct, 
        end_value = config["learning_rate"]*0.1
        )
    grad_transforms = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.1))
    return nnx.Optimizer(model, grad_transforms, wrt= nnx.Param)


@nnx.jit
def training_step(model, optimizer, In, Out, mask, aux_factor):
    loss, grad = nnx.value_and_grad(calc_total_loss)(model, In, Out, mask, aux_factor)
    optimizer.update(model, grad)
    return loss


class MoE(nnx.Module):
    def __init__(self, config, rngs):
        self.E, self.k = config["E"], config["k"]
        self.router = nnx.Linear(config["D"], self.E, use_bias=False, rngs=rngs)
        self.w_up = nnx.Param(jax.random.normal(rngs.params(), (self.E, config["D"], config["F"])) / jnp.sqrt(config["D"]))
        self.w_down = nnx.Param(jax.random.normal(rngs.params(), (self.E, config["F"], config["D"])) / jnp.sqrt(config["F"]))

    def __call__(self, x):
        B, T, D = x.shape
        x_flattened = x.reshape(-1, D) #(B*T, D)
        M = x_flattened.shape[0]

        probs = jax.nn.softmax(self.router(x_flattened), axis=-1)
        gates, experts = jax.lax.top_k(probs, self.k) #(B*T, K), (B*T, K)
        if self.k > 1:
            gates = gates / gates.sum(-1, keepdims=True) #(B*T, K)

        experts_flattened = experts.reshape(-1)
        order = jnp.argsort(experts_flattened)
        x_sorted = x_flattened[order // self.k]
        counts = jnp.bincount(experts_flattened, length=self.E) #(B*k, D)
        group_sizes = counts.astype(jnp.int32)

        h = jax.lax.ragged_dot(x_sorted, self.w_up[...], group_sizes) #(B*k, F)
        y_sorted = jax.lax.ragged_dot(jax.nn.gelu(h), self.w_down[...], group_sizes) #(B*k, D)

        y = y_sorted[jnp.argsort(order)].reshape(M, self.k, D)
        out = (y * gates[..., None]).sum(axis=1) #(B, D)
        real_split = counts/experts_flattened.size
        aux_loss = self.E * jnp.sum(real_split * probs.mean(axis=0))
        return out.reshape(B, T, D), aux_loss



if __name__ == "__main__":
    config = Config().config
    Tblock = TransformerBlock(config, nnx.Rngs(0))
    n = sum(p.size for p in jax.tree.leaves(nnx.state(Tblock, nnx.Param)))
    print(n)

    # x = jax.random.normal(jax.random.key(1), (2, 11, config["D"]))
    # y1 = Tblock(x)
    # print(y1.shape == x.shape)

    # x2 = x.at[:, 8].set(0.0)
    # y2 = Tblock(x2)
    # assert jnp.allclose(y1[:, :8], y2[:, :8], atol=1e-5)  
    # assert not jnp.allclose(y1[:, 8:], y2[:, 8:])
    # print("block OK")