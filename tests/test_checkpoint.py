import json

import torch

from ullm.config import ModelConfig
from ullm.model import GPT

CONFIG = ModelConfig(
    n_layers=2, d_model=32, n_heads=4, mlp_hidden=96, context_length=16, vocab_size=64
)


def test_save_writes_weights_and_config(tmp_path):
    GPT(CONFIG).save(tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "model.safetensors").is_file()
    config = json.loads((tmp_path / "ckpt" / "config.json").read_text())
    assert config == {
        "n_layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "mlp_hidden": 96,
        "context_length": 16,
        "vocab_size": 64,
    }


def test_loaded_model_gives_identical_logits(tmp_path):
    torch.manual_seed(0)
    model = GPT(CONFIG)
    model.save(tmp_path)

    loaded = GPT.load(tmp_path)
    assert loaded.config == CONFIG
    assert loaded.head.weight is loaded.tok_emb.weight
    idx = torch.randint(0, CONFIG.vocab_size, (2, 10))
    with torch.no_grad():
        torch.testing.assert_close(loaded(idx), model(idx), rtol=0, atol=0)


def test_load_onto_device(tmp_path):
    GPT(CONFIG).save(tmp_path)
    loaded = GPT.load(tmp_path, device="cpu")
    assert all(p.device.type == "cpu" for p in loaded.parameters())
    assert loaded.rope_cos.device.type == "cpu"
