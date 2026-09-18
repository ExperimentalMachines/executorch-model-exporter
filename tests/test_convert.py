import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from pipeline import convert  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB = 8, 2, 1, 4, 16, 10


def hf_permute(w, n_heads):
    """What HF's convert_llama_weights_to_hf.py does to Meta q/k weights."""
    dim1, dim2 = w.shape
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def tiny_config(layers=2, tied=True):
    return {
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "num_hidden_layers": layers,
        "tie_word_embeddings": tied,
    }


def tiny_checkpoint(layers=2, qk_norm=False, qkv_bias=False, tied=True):
    g = torch.Generator().manual_seed(0)

    def t(*shape):
        return torch.randn(*shape, generator=g)

    sd = {"model.embed_tokens.weight": t(VOCAB, HIDDEN), "model.norm.weight": t(HIDDEN)}
    if not tied:
        sd["lm_head.weight"] = t(VOCAB, HIDDEN)
    for i in range(layers):
        p = f"model.layers.{i}."
        sd |= {
            p + "self_attn.q_proj.weight": t(HEADS * HEAD_DIM, HIDDEN),
            p + "self_attn.k_proj.weight": t(KV_HEADS * HEAD_DIM, HIDDEN),
            p + "self_attn.v_proj.weight": t(KV_HEADS * HEAD_DIM, HIDDEN),
            p + "self_attn.o_proj.weight": t(HIDDEN, HEADS * HEAD_DIM),
            p + "input_layernorm.weight": t(HIDDEN),
            p + "post_attention_layernorm.weight": t(HIDDEN),
            p + "mlp.gate_proj.weight": t(FFN, HIDDEN),
            p + "mlp.up_proj.weight": t(FFN, HIDDEN),
            p + "mlp.down_proj.weight": t(HIDDEN, FFN),
        }
        if qk_norm:
            sd[p + "self_attn.q_norm.weight"] = t(HEAD_DIM)
            sd[p + "self_attn.k_norm.weight"] = t(HEAD_DIM)
        if qkv_bias:
            sd[p + "self_attn.q_proj.bias"] = t(HEADS * HEAD_DIM)
            sd[p + "self_attn.k_proj.bias"] = t(KV_HEADS * HEAD_DIM)
            sd[p + "self_attn.v_proj.bias"] = t(KV_HEADS * HEAD_DIM)
    return sd


def test_unpermute_inverts_hf_permute():
    w = torch.randn(HEADS * HEAD_DIM, HIDDEN)
    assert torch.equal(convert.unpermute(hf_permute(w, HEADS), HEADS), w)
    k = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN)
    assert torch.equal(convert.unpermute(hf_permute(k, KV_HEADS), KV_HEADS), k)


def test_map_key():
    assert convert.map_key("model.layers.3.mlp.gate_proj.weight", "llama") == "layers.3.feed_forward.w1.weight"
    assert convert.map_key("model.layers.0.self_attn.q_norm.weight", "qwen3") == "layers.0.attention.q_norm_fn.weight"
    assert convert.map_key("model.layers.0.self_attn.k_proj.bias", "qwen2") == "layers.0.attention.wk.bias"
    assert convert.map_key("model.layers.0.self_attn.rotary_emb.inv_freq", "llama") is None
    with pytest.raises(KeyError):
        convert.map_key("model.layers.0.self_attn.q_norm.weight", "llama")
    with pytest.raises(KeyError):
        convert.map_key("model.something_new.weight", "qwen3")


def test_convert_qwen3_ties_output_to_embeddings(tmp_path):
    sd = tiny_checkpoint(qk_norm=True)
    safetensors_torch.save_file(sd, tmp_path / "model.safetensors")
    summary = convert.convert(tmp_path, tmp_path / "out" / "consolidated.pth", "qwen3", tiny_config())
    out = torch.load(tmp_path / "out" / "consolidated.pth")
    assert summary["tied_embeddings"] is True
    assert torch.equal(out["output.weight"], sd["model.embed_tokens.weight"])
    # Qwen converters do not permute: HF RoPE is used at export.
    assert torch.equal(out["layers.1.attention.wq.weight"], sd["model.layers.1.self_attn.q_proj.weight"])
    assert torch.equal(out["layers.0.attention.k_norm_fn.weight"], sd["model.layers.0.self_attn.k_norm.weight"])
    assert summary["tensors"] == 2 * (9 + 2) + 3


def test_convert_llama_unpermutes_q_and_k(tmp_path):
    sd = tiny_checkpoint(tied=False)
    meta_q = torch.randn(HEADS * HEAD_DIM, HIDDEN)
    meta_k = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN)
    sd["model.layers.0.self_attn.q_proj.weight"] = hf_permute(meta_q, HEADS)
    sd["model.layers.0.self_attn.k_proj.weight"] = hf_permute(meta_k, KV_HEADS)
    safetensors_torch.save_file(sd, tmp_path / "model.safetensors")
    summary = convert.convert(tmp_path, tmp_path / "consolidated.pth", "llama", tiny_config(tied=False))
    out = torch.load(tmp_path / "consolidated.pth")
    assert summary["tied_embeddings"] is False
    assert torch.equal(out["layers.0.attention.wq.weight"], meta_q)
    assert torch.equal(out["layers.0.attention.wk.weight"], meta_k)
    assert torch.equal(out["output.weight"], sd["lm_head.weight"])


def test_sharded_checkpoint(tmp_path):
    sd = tiny_checkpoint(qkv_bias=True)
    keys = sorted(sd)
    half = len(keys) // 2
    shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
    weight_map = {}
    for name, shard_keys in shards.items():
        safetensors_torch.save_file({k: sd[k] for k in shard_keys}, tmp_path / name)
        weight_map |= {k: name for k in shard_keys}
    import json

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    summary = convert.convert(tmp_path, tmp_path / "c.pth", "qwen2", tiny_config())
    assert summary["tensors"] == 2 * (9 + 3) + 3


def test_missing_layer_tensors_fail(tmp_path):
    sd = tiny_checkpoint(qk_norm=True)
    del sd["model.layers.1.mlp.up_proj.weight"]
    safetensors_torch.save_file(sd, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="expected 2 layers"):
        convert.convert(tmp_path, tmp_path / "c.pth", "qwen3", tiny_config())


def test_checkpoint_name_that_would_trigger_prequantized_loading(tmp_path):
    with pytest.raises(ValueError, match="pre-quantized"):
        convert.convert(tmp_path, tmp_path / "model-8da4w.pth", "qwen3", tiny_config())


def test_lfm2_keys_map_to_executorchs_names():
    m = lambda k: convert.map_key(k, "lfm2")  # noqa: E731
    assert m("model.embed_tokens.weight") == "tok_embeddings.weight"
    # LFM2 keeps the final norm under its own name and has no lm_head.
    assert m("model.embedding_norm.weight") == "norm.weight"
    # It names the attention output out_proj and the QK norms *_layernorm.
    assert m("model.layers.2.self_attn.out_proj.weight") == "layers.2.attention.wo.weight"
    assert m("model.layers.2.self_attn.q_layernorm.weight") == "layers.2.attention.q_norm_fn.weight"
    # The block norm is operator_norm, where the other families say input_layernorm.
    assert m("model.layers.0.operator_norm.weight") == "layers.0.attention_norm.weight"
    # The feed-forward and the convolution already carry ExecuTorch's names.
    assert m("model.layers.0.feed_forward.w1.weight") == "layers.0.feed_forward.w1.weight"
    assert m("model.layers.0.conv.conv.weight") == "layers.0.conv.conv.weight"


def test_lfm2_does_not_borrow_the_other_families_names():
    # o_proj and input_layernorm belong to the common map, which LFM2 replaces rather than
    # extends; accepting them would silently convert a checkpoint that is not LFM2.
    for key in ("model.layers.0.self_attn.o_proj.weight", "model.layers.0.input_layernorm.weight"):
        with pytest.raises(KeyError):
            convert.map_key(key, "lfm2")


def test_the_fused_convolution_projection_splits_into_three():
    target, parts = convert.SPLIT_THREE["lfm2"]
    assert target == "conv.in_proj.weight"
    assert parts == ("conv.B_proj.weight", "conv.C_proj.weight", "conv.x_proj.weight")
