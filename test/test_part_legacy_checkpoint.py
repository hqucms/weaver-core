"""Unit tests for loading weaver-core < 0.5 ParticleTransformer checkpoints.

Before 0.5, ``Block`` used ``nn.MultiheadAttention`` (state-dict keys
``attn.in_proj_weight``) and applied ``c_attn`` via
``einsum("tbhd,h->tbdh")``, which left the attention output in
(head_dim, num_heads) channel order. The current ``Block`` keeps
(num_heads, head_dim) order, so a legacy checkpoint loads without any key error
but gives wrong outputs unless ``Block._load_from_state_dict`` converts it.

The reference below is the 0.4.17 ``Block`` forward pass (seq-first, as it was).
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from weaver.nn.model.ParticleTransformer import Block


def _legacy_state_dict(block):
    """State dict of ``block`` rewritten the way a 0.4.x checkpoint stores it."""
    sd = {}
    for k, v in block.state_dict().items():
        k = k.replace("attn.in_proj.weight", "attn.in_proj_weight").replace("attn.in_proj.bias", "attn.in_proj_bias")
        sd[k] = v.clone()
    return sd


def _legacy_forward(sd, x, x_cls=None, padding_mask=None, attn_mask=None, num_heads=8):
    """0.4.17 ``Block.forward`` on a legacy state dict; batch-first in and out."""
    embed_dim = x.size(-1)
    head_dim = embed_dim // num_heads
    mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0).eval()
    mha.load_state_dict(
        {
            "in_proj_weight": sd["attn.in_proj_weight"],
            "in_proj_bias": sd["attn.in_proj_bias"],
            "out_proj.weight": sd["attn.out_proj.weight"],
            "out_proj.bias": sd["attn.out_proj.bias"],
        }
    )

    def ln(t, name):
        return F.layer_norm(t, t.shape[-1:], sd[name + ".weight"], sd[name + ".bias"])

    x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
    if x_cls is not None:
        x_cls = x_cls.transpose(0, 1)
        padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
        residual = x_cls
        u = ln(torch.cat((x_cls, x), dim=0), "pre_attn_norm")
        x = mha(x_cls, u, u, key_padding_mask=padding_mask)[0]
    else:
        residual = x
        x = ln(x, "pre_attn_norm")
        x = mha(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]

    tgt_len = x.size(0)
    x = x.view(tgt_len, -1, num_heads, head_dim)
    x = torch.einsum("tbhd,h->tbdh", x, sd["c_attn"])
    x = x.reshape(tgt_len, -1, embed_dim)
    x = ln(x, "post_attn_norm")
    x = x + residual

    residual = x
    x = F.gelu(F.linear(ln(x, "pre_fc_norm"), sd["fc1.weight"], sd["fc1.bias"]))
    x = F.linear(ln(x, "post_fc_norm"), sd["fc2.weight"], sd["fc2.bias"])
    x = x + sd["w_resid"] * residual
    return x.transpose(0, 1)


class TestLegacyCheckpoint(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.num_heads, self.embed_dim = 8, 128
        src = Block(embed_dim=self.embed_dim, num_heads=self.num_heads).eval()
        with torch.no_grad():
            for p in src.parameters():
                p.add_(0.1 * torch.randn_like(p))  # non-trivial c_attn, norms and biases
        self.legacy_sd = _legacy_state_dict(src)

        bsz, seq_len = 4, 16
        self.x = torch.randn(bsz, seq_len, self.embed_dim)
        self.padding_mask = torch.zeros(bsz, seq_len, dtype=torch.bool)
        self.padding_mask[:, 12:] = True
        self.attn_mask = torch.randn(bsz * self.num_heads, seq_len, seq_len)
        self.x_cls = torch.randn(bsz, 1, self.embed_dim)

    def _load(self, sd):
        block = Block(embed_dim=self.embed_dim, num_heads=self.num_heads).eval()
        block.load_state_dict({k: v.clone() for k, v in sd.items()})
        return block

    def test_particle_attention_matches_legacy(self):
        block = self._load(self.legacy_sd)
        attn_mask = self.attn_mask.view(-1, self.num_heads, *self.attn_mask.shape[1:])
        with torch.no_grad():
            ref = _legacy_forward(self.legacy_sd, self.x, padding_mask=self.padding_mask, attn_mask=self.attn_mask)
            out = block(self.x, padding_mask=self.padding_mask, attn_mask=attn_mask)
        valid = ~self.padding_mask
        torch.testing.assert_close(out[valid], ref[valid], atol=1e-5, rtol=1e-5)

    def test_class_attention_matches_legacy(self):
        block = self._load(self.legacy_sd)
        with torch.no_grad():
            ref = _legacy_forward(self.legacy_sd, self.x, x_cls=self.x_cls, padding_mask=self.padding_mask)
            out = block(self.x, x_cls=self.x_cls, padding_mask=self.padding_mask)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_current_checkpoint_is_untouched(self):
        block = self._load(self.legacy_sd)
        sd = block.state_dict()
        self.assertIn("attn.in_proj.weight", sd)
        reloaded = self._load(sd)
        for k, v in sd.items():
            torch.testing.assert_close(reloaded.state_dict()[k], v, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
