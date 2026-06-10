"""MDLM — Masked Diffusion Language Model for promoter sequences.

Discrete absorbing-state diffusion over {A, T, G, C, [MASK]}:
forward corrupts tokens to [MASK]; the Transformer learns to un-mask.
Pretraining is unconditional (time only); fine-tuning adds an expression
condition with classifier-free guidance.

Ref: Sahoo et al., MDLM, NeurIPS 2024.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.genome_dataset import MASK_TOKEN_ID, VOCAB_SIZE


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of continuous t ∈ [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] → [B, dim]
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConditionEmbedding(nn.Module):
    """Time (+ optional expression) → hidden vector for AdaLN modulation.

    `n_cond_classes > 0` → discrete class embedding; else → continuous MLP.
    `null_cond` is the DiT-style learnable unconditional token used for CFG.
    """

    def __init__(self, hidden_dim, has_expression_cond=False, n_cond_classes=0):
        super().__init__()
        self.has_expression_cond = has_expression_cond
        self.n_cond_classes = n_cond_classes

        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        if has_expression_cond:
            if n_cond_classes > 0:
                self.cond_emb_layer = nn.Embedding(n_cond_classes, hidden_dim)
                # Zero-init so fine-tuning starts from the pretrained unconditional model.
                nn.init.zeros_(self.cond_emb_layer.weight)
            else:
                self.cond_mlp = nn.Sequential(
                    nn.Linear(1, hidden_dim * 4),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
            self.null_cond = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, t, cond=None, cond_drop_mask=None, force_null_cond=False):
        h = self.time_mlp(self.time_emb(t))
        if not self.has_expression_cond:
            return h

        if force_null_cond:
            return h + self.null_cond

        if cond is not None:
            if self.n_cond_classes > 0:
                cond_emb = self.cond_emb_layer(cond)          # [B, D]
            else:
                cond_emb = self.cond_mlp(cond[:, None])       # [B, D]

            if cond_drop_mask is not None:
                null = self.null_cond.unsqueeze(0).expand_as(cond_emb)
                cond_emb = torch.where(cond_drop_mask[:, None], null, cond_emb)
            h = h + cond_emb

        return h


class AdaLayerNorm(nn.Module):
    """DiT-style AdaLN: LN(x) * (1 + scale(c)) + shift(c)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim * 2)
        # Zero-init makes every AdaLN start as an identity LN — the AdaLN-Zero trick.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        # x: [B, L, D], cond: [B, D]
        x = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class TransformerBlock(nn.Module):
    """Pre-AdaLN self-attn + FFN."""

    def __init__(self, hidden_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = AdaLayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond):
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h

        h = self.norm2(x, cond)
        x = x + self.ffn(h)
        return x


class DNATransformer(nn.Module):
    """Token + position emb → N × AdaLN Transformer block → vocab logits."""

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        seq_len=81,
        hidden_dim=256,
        n_layers=8,
        n_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        has_expression_cond=False,
        n_cond_classes=0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.cond_emb = ConditionEmbedding(hidden_dim, has_expression_cond, n_cond_classes)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, t, cond=None, cond_drop_mask=None, force_null_cond=False):
        # x: [B, L] token ids (may contain MASK); returns logits [B, L, V]
        _, L = x.shape
        pos_ids = torch.arange(L, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos_ids)
        c = self.cond_emb(t, cond, cond_drop_mask, force_null_cond)
        for block in self.blocks:
            h = block(h, c)
        return self.output_head(self.final_norm(h))


class MDLM(nn.Module):
    """Wraps the noise schedule, forward corruption, loss, and sampler."""

    def __init__(
        self,
        seq_len=81,
        hidden_dim=256,
        n_layers=8,
        n_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        has_expression_cond=False,
        n_cond_classes=0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_cond_classes = n_cond_classes

        self.transformer = DNATransformer(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            has_expression_cond=has_expression_cond,
            n_cond_classes=n_cond_classes,
        )

    def noise_schedule(self, t):
        # Cosine mask schedule: 0 at t=0, 1 at t=1.
        return 1.0 - torch.cos(t * math.pi / 2)

    def forward_diffusion(self, x_0, t):
        """Replace each position with MASK independently with prob `noise_schedule(t)`."""
        mask_prob = self.noise_schedule(t)                     # [B]
        mask = torch.rand_like(x_0, dtype=torch.float) < mask_prob[:, None]
        x_t = x_0.clone()
        x_t[mask] = MASK_TOKEN_ID
        return x_t, mask

    def compute_loss(self, x_0, t=None, cond=None, cfg_dropout=0.0,
                     return_metrics=False):
        """Cross-entropy over masked positions only.

        `cfg_dropout`: per-sample prob of replacing `cond` with `null_cond`
        during training — this is what makes CFG work at sample time.
        """
        B, _ = x_0.shape
        device = x_0.device

        if t is None:
            # Clamp away from the boundaries: t=0 has no mask to learn from,
            # t=1 leaks zero signal from the input.
            t = torch.rand(B, device=device) * 0.98 + 0.01

        x_t, mask = self.forward_diffusion(x_0, t)

        cond_drop_mask = None
        if cfg_dropout > 0 and cond is not None:
            cond_drop_mask = torch.rand(B, device=device) < cfg_dropout

        logits = self.transformer(x_t, t, cond, cond_drop_mask)  # [B, L, V]

        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return (loss, {}) if return_metrics else loss

        logits_masked = logits[mask]        # [N_masked, V]
        targets_masked = x_0[mask]          # [N_masked]
        loss = F.cross_entropy(logits_masked, targets_masked)

        if return_metrics:
            with torch.no_grad():
                preds = logits_masked.argmax(dim=-1)
                correct = (preds == targets_masked)
                metrics = {
                    'accuracy': correct.float().mean().item(),
                    'perplexity': loss.exp().item(),
                    'n_masked': mask.sum().item(),
                    'mask_ratio': mask.float().mean().item(),
                }
                for base_id, base_name in enumerate(['A', 'T', 'G', 'C']):
                    bm = (targets_masked == base_id)
                    if bm.sum() > 0:
                        metrics[f'acc_{base_name}'] = correct[bm].float().mean().item()
            return loss, metrics

        return loss

    @torch.no_grad()
    def sample(self, batch_size, cond=None, guidance_scale=1.0, steps=1000,
               device='cuda', x_init=None, fixed_mask=None):
        """Ancestral sampler: start from all-MASK (or x_init) and walk t: 1 → 0.

        `x_init` + `fixed_mask`: inpainting — positions where `fixed_mask` is
        True are re-pinned to `x_init` every step.
        """
        self.eval()
        L = self.seq_len

        if x_init is not None:
            x = x_init.clone().to(device)
        else:
            x = torch.full((batch_size, L), MASK_TOKEN_ID, dtype=torch.long, device=device)

        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]

            mask_prob_now = self.noise_schedule(torch.tensor([t_now], device=device)).item()
            mask_prob_next = self.noise_schedule(torch.tensor([t_next], device=device)).item()

            is_masked = (x == MASK_TOKEN_ID)
            if not is_masked.any():
                break

            t_batch = torch.full((batch_size,), t_now, device=device)

            if guidance_scale != 1.0 and cond is not None:
                # CFG: logits = uncond + s * (cond - uncond)
                logits_cond = self.transformer(x, t_batch, cond=cond)
                logits_uncond = self.transformer(x, t_batch, force_null_cond=True)
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = self.transformer(x, t_batch, cond=cond)

            # Restrict to ATGC — we never want to sample the MASK token back in.
            logits_dna = logits[:, :, :4]
            probs = F.softmax(logits_dna, dim=-1)
            pred_tokens = torch.multinomial(
                probs.view(-1, 4), num_samples=1,
            ).view(batch_size, L)

            # Expected fraction of still-masked positions to reveal this step.
            unmask_prob = 1.0 - mask_prob_next / mask_prob_now if mask_prob_now > 0 else 1.0
            unmask_rand = torch.rand_like(x, dtype=torch.float)
            to_unmask = is_masked & (unmask_rand < unmask_prob)
            x[to_unmask] = pred_tokens[to_unmask]

            if fixed_mask is not None and x_init is not None:
                x[fixed_mask] = x_init[fixed_mask]

        # Anything still masked at t=0: do one deterministic-ish final pass.
        remaining_mask = (x == MASK_TOKEN_ID)
        if remaining_mask.any():
            t_batch = torch.full((batch_size,), 0.0, device=device)
            logits = self.transformer(x, t_batch, cond=cond)
            probs = F.softmax(logits[:, :, :4], dim=-1)
            pred_tokens = torch.multinomial(
                probs.view(-1, 4), num_samples=1,
            ).view(batch_size, L)
            x[remaining_mask] = pred_tokens[remaining_mask]

        return x
