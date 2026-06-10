"""Sampling utilities: batched conditional generation and inpainting."""

import torch

from src.data.genome_dataset import decode_sequence, MASK_TOKEN_ID, BASE_TO_ID


def generate_sequences(model, cond_value, guidance_scale, n_samples,
                       steps, batch_size, device, discrete=False):
    """Batch-sample `n_samples` sequences under a fixed cond."""
    model.eval()
    sequences = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in range(n_batches):
        bs = min(batch_size, n_samples - len(sequences))
        if discrete:
            cond = torch.full((bs,), cond_value, dtype=torch.long, device=device)
        else:
            cond = torch.full((bs,), cond_value, device=device)
        samples = model.sample(
            batch_size=bs,
            cond=cond,
            guidance_scale=guidance_scale,
            steps=steps,
            device=device,
        )
        for j in range(bs):
            sequences.append(decode_sequence(samples[j]))

    return sequences


def generate_inpainting(model, template_seq, fixed_positions, cond_value,
                        guidance_scale, n_samples, steps, batch_size,
                        device, discrete=False):
    """Inpainting: fix positions in `template_seq`, sample the rest."""
    model.eval()
    seq_len = len(template_seq)
    sequences = []

    x_template = torch.full((1, seq_len), MASK_TOKEN_ID, dtype=torch.long)
    mask_template = torch.zeros((1, seq_len), dtype=torch.bool)

    for pos in fixed_positions:
        base = template_seq[pos].upper()
        if base in BASE_TO_ID:
            x_template[0, pos] = BASE_TO_ID[base]
            mask_template[0, pos] = True

    n_fixed = mask_template.sum().item()
    n_to_fill = seq_len - n_fixed
    print(f"  inpainting: {n_fixed} fixed, {n_to_fill} to sample")

    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in range(n_batches):
        bs = min(batch_size, n_samples - len(sequences))

        x_init = x_template.expand(bs, -1).clone().to(device)
        fixed_mask = mask_template.expand(bs, -1).clone().to(device)

        if discrete:
            cond = torch.full((bs,), cond_value, dtype=torch.long, device=device)
        else:
            cond = torch.full((bs,), cond_value, device=device)

        samples = model.sample(
            batch_size=bs, cond=cond, guidance_scale=guidance_scale,
            steps=steps, device=device,
            x_init=x_init, fixed_mask=fixed_mask,
        )
        for j in range(bs):
            sequences.append(decode_sequence(samples[j]))

    return sequences


def parse_template(template_str):
    """Template string -> fixed position list. ATGC = fixed; -/N/. = sample."""
    fixed_positions = []
    for i, ch in enumerate(template_str.upper()):
        if ch in 'ATGC':
            fixed_positions.append(i)
    return fixed_positions
