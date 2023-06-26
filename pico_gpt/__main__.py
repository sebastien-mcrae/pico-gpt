import fire
import numpy as np
from tqdm import tqdm

from pico_gpt.utils import gelu, linear, load_encoder_hparams_and_params, softmax


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray
) -> np.ndarray:  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def multi_head_attention(
    x: np.ndarray, c_attn: dict, c_proj: dict, n_head: int, kv_cache: list | None = None
) -> tuple[np.ndarray, list]:  # [n_seq, n_embd] -> [n_seq, n_embd]
    # QKV projection.
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3 * n_embd]
    # Split into QKV.
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3 * n_embd] -> [3, n_seq, n_embd]

    if kv_cache is not None:
        new_q, new_k, new_v = qkv
        old_k, old_v = kv_cache
        k = np.vstack(
            [old_k, new_k]
        )  # k = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        v = np.vstack(
            [old_v, new_v]
        )  # v = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        qkv = [new_q, k, v]
        causal_mask = np.zeros((1, k.shape[0]))
    else:
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    current_cache = [qkv[1], qkv[2]]

    # Split into heads.
    qkv_heads = list(
        map(lambda x: np.split(x, n_head, axis=-1), qkv)
    )  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd / n_head]
    # Perform attention over each head.
    out_heads = [
        attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)
    ]  # [3, n_head, n_seq, n_embd / n_head] -> [n_head, n_seq, n_embd / n_head]
    # Merge heads.
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd / n_head] -> [n_seq, n_embd]
    # Out projection.
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x, current_cache


def ffn(
    x: np.ndarray, c_fc: dict, c_proj: dict
) -> np.ndarray:  # [n_seq, n_embd] -> [n_seq, n_embd]
    # Project up (from n_emb to 4 * n_emb).
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4 * n_embd]
    # Project back down.
    return linear(a, **c_proj)  # [n_seq, 4 * n_embd] -> [n_seq, n_embd]


def layer_norm(
    x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(
        variance + eps
    )  # Normalize x to have mean=0 and var=1 over last axis.
    return g * x + b  # Scale and offset with gamma/beta params.


def transformer_block(
    x: np.ndarray,
    mlp: dict,
    attn: dict,
    ln_1: dict,
    ln_2: dict,
    n_head: int,
    kv_cache: list | None = None,
) -> tuple[np.ndarray, list]:  # [n_seq, n_embd] -> [n_seq, n_embd]
    # Multi-head causal self attention.
    attn_out, kv_cache_updated = multi_head_attention(
        layer_norm(x, **ln_1), **attn, n_head=n_head, kv_cache=kv_cache
    )
    x += attn_out  # [n_seq, n_embd] -> [n_seq, n_embd]
    # Position-wise feed forward network.
    x += ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x, kv_cache_updated


def gpt_2(
    inputs: list[int],
    wte: np.ndarray,  # Token embeddings.
    wpe: np.ndarray,  # Positional encodings.
    blocks: list[dict],
    ln_f: dict,
    n_head: int,
    kv_cache: list | None = None,
) -> tuple[np.ndarray, list]:  # [n_seq] -> [n_seq, n_vocab]
    if kv_cache is None:
        kv_cache = [None] * len(blocks)
        x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    else:
        x = wte[[inputs[-1]]] + wpe[[len(inputs) - 1]]

    # Forward pass through n_layer transformer blocks.
    new_kv_cache = []
    for block, kv_cache_block in zip(blocks, kv_cache):
        x, updated_cache = transformer_block(
            x, **block, n_head=n_head, kv_cache=kv_cache_block
        )  # [n_seq, n_embd] -> [n_seq, n_embd]
        new_kv_cache.append(updated_cache)
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    # Language Modeling head: projection to vocab (logits).
    return x @ wte.T, new_kv_cache  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(
    inputs: list[int], params: dict, n_head: int, n_tokens_to_generate: int
) -> list[int]:
    kv_cache = None
    generated_tokens = []
    for _ in tqdm(
        range(n_tokens_to_generate), f"Generating {n_tokens_to_generate} tokens..."
    ):
        logits, kv_cache = gpt_2(
            inputs, **params, n_head=n_head, kv_cache=kv_cache
        )  # Forward pass.
        next_id = np.argmax(logits[-1])  # Greedy sampling.
        inputs.append(int(next_id))  # Append prediction to input.
        generated_tokens.append((int(next_id)))
    return generated_tokens


def main(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: str = "124M",
    models_dir: str = "models",
) -> str:
    # Load encoder, hparams, and params from the released OpenAI GPT-2 files.
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    # Encode the input string using the BPE tokenizer.
    input_ids = encoder.encode(prompt)
    # Make sure we are not surpassing the max sequence length of our model.
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    # Generate output IDs.
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    # Decode the IDs back into a string.
    output_text = encoder.decode(output_ids)
    return f"\nPrompt: {prompt}\nLLM output: {output_text}"


if __name__ == "__main__":
    fire.Fire(main)
