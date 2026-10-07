# import time
from transformer import Transformer, Optimizer, training_step, jitted_loss
from config import Config, make_mask
from flax import nnx
import jax.numpy as jnp
import jax
# import numpy as np
from data_gen import gen_data
from tokenizer import Tokenizer
from checkpoint import save
config = Config().config

# t0 = time.time()
# for step in range(300):
#     loss = training_step(model, optimizer, x, y, mask)
#     if step % 50 == 0:
#         print(step, float(loss), f"{time.time() - t0:.1f}s")

def train_model(config, train, test, epochs = 2):
    model = Transformer(config, nnx.Rngs(config["seed"]))
    epoch_steps = len(train)//config["B"]
    total_steps =epoch_steps*epochs
    optimizer = Optimizer(model, config, stepct=total_steps)
    mask = make_mask(config)
    train_jax = jnp.array(train)
    test_jax = jnp.array(test[:2048])
    key = jax.random.key(config["seed"])
    step=0
    for ep in range(epochs):
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, len(train_jax))
        for i in range(epoch_steps):
            batch = train_jax[perm[i*config["B"]:(i+1)*config["B"]]]
            x, y = batch[:, :-1], batch[:, 1:]
            loss = training_step(model, optimizer, x, y, mask, config["aux_factor"])
            step+=1
            if(step%500==0):
                test_loss = jitted_loss(model, test_jax[:,:-1], test_jax[:,1:], mask)
                print(f"epoch {ep} step {step}/{total_steps} "
                      f"train {float(loss):.4f} test {float(test_loss):.4f}")
    return model, optimizer

# model, optimizer = train_model(config, train, test, 2)

if __name__ == "__main__":
    train, test = gen_data(config, Tokenizer())
    model, optimizer = train_model(config, train, test, 2)
    save(model, optimizer)