import os
import orbax.checkpoint as ocp
from flax import nnx
from transformer import Transformer

CKPT_DIR = os.path.abspath("checkpoints")   # on Colab: "/content/drive/MyDrive/jax-addition/checkpoints"

def save(model, optimizer, name="addition"):
    ckpt = ocp.StandardCheckpointer()
    _, m_state = nnx.split(model)
    _, o_state = nnx.split(optimizer)
    ckpt.save(os.path.join(CKPT_DIR, name, "model"), m_state, force=True)
    ckpt.save(os.path.join(CKPT_DIR, name, "optimizer"), o_state, force=True)
    ckpt.wait_until_finished()

def load_model(config, name="addition"):
    abstract = nnx.eval_shape(lambda: Transformer(config, nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract)
    state = ocp.StandardCheckpointer().restore(
        os.path.join(CKPT_DIR, name, "model"), abstract_state)
    return nnx.merge(graphdef, state)