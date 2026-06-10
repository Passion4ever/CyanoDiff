"""Training/run helpers shared by pretrain and finetune scripts."""

import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data.genome_dataset import decode_sequence


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps,
                                    min_lr_ratio=0.0):
    """Cosine schedule with warmup, floored at `min_lr_ratio * peak_lr`.
    min_lr_ratio=0.0 is a plain cosine decay (no floor)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_run_dir(base_dir, run_name='run'):
    """Layout: {base_dir}/{run_name}/{MMDD_HHMMSS}/"""
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    run_dir = Path(base_dir) / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(out_dir, name='train', log_file='train.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(Path(out_dir) / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def validate(model, val_loader, device):
    """Unconditional validation: mean loss + accuracy."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x_0 = batch.to(device)
            loss, metrics = model.compute_loss(x_0, return_metrics=True)
            total_loss += loss.item()
            total_acc += metrics.get('accuracy', 0.0)
            n_batches += 1

    model.train()
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def validate_conditional(model, val_loader, device):
    """Per-class loss/acc breakdown — high-expression samples are sparse and
    worth watching separately."""
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    base_acc_sum = {b: 0.0 for b in ['A', 'T', 'G', 'C']}
    base_acc_count = {b: 0 for b in ['A', 'T', 'G', 'C']}
    bin_names = ['low', 'mid', 'high']
    bin_loss = {name: 0.0 for name in bin_names}
    bin_acc = {name: 0.0 for name in bin_names}
    bin_count = {name: 0 for name in bin_names}

    with torch.no_grad():
        for seq, expr in val_loader:
            seq = seq.to(device)
            expr = expr.to(device)
            # No CFG dropout at validation.
            loss, metrics = model.compute_loss(
                seq, cond=expr, cfg_dropout=0.0, return_metrics=True,
            )
            total_loss += loss.item()
            total_acc += metrics.get('accuracy', 0.0)
            n_batches += 1

            for base_name in ['A', 'T', 'G', 'C']:
                key = f'acc_{base_name}'
                if key in metrics:
                    base_acc_sum[base_name] += metrics[key]
                    base_acc_count[base_name] += 1

            is_discrete = expr.dtype in (torch.long, torch.int)
            for i, name in enumerate(bin_names):
                if is_discrete:
                    mask = (expr == i)
                else:
                    # Continuous-cond fallback buckets.
                    edges = [0.0, 0.2, 0.4, 1.01]
                    mask = (expr >= edges[i]) & (expr < edges[i + 1])
                if mask.sum() > 0:
                    loss_bin, metrics_bin = model.compute_loss(
                        seq[mask], cond=expr[mask], cfg_dropout=0.0,
                        return_metrics=True,
                    )
                    bin_loss[name] += loss_bin.item()
                    bin_acc[name] += metrics_bin.get('accuracy', 0.0)
                    bin_count[name] += 1

    model.train()

    result = {
        'val_loss': total_loss / max(1, n_batches),
        'val_acc': total_acc / max(1, n_batches),
    }
    for base_name in ['A', 'T', 'G', 'C']:
        if base_acc_count[base_name] > 0:
            result[f'val_acc_{base_name}'] = base_acc_sum[base_name] / base_acc_count[base_name]
    for name in bin_names:
        if bin_count[name] > 0:
            result[f'val_loss_{name}'] = bin_loss[name] / bin_count[name]
            result[f'val_acc_{name}'] = bin_acc[name] / bin_count[name]

    return result


def evaluate_generation(model, device, steps=200, n_samples=16, n_cond_classes=0):
    """Mid-training generation probe: sample under each condition and check
    GC spread / motif recovery to see whether CFG is doing anything."""
    model.eval()

    if n_cond_classes > 0:
        target_conds = list(range(n_cond_classes))
        cond_labels = [f'class{c}' for c in target_conds]
    else:
        target_conds = [0.1, 0.3, 0.5, 0.7]
        cond_labels = [f'c{c:.1f}' for c in target_conds]

    guidance_scale = 2.0
    motif = 'GCGATCGC'

    results = {}
    sample_table = []

    for cond_val, label in zip(target_conds, cond_labels):
        if n_cond_classes > 0:
            cond = torch.full((n_samples,), cond_val, dtype=torch.long, device=device)
        else:
            cond = torch.full((n_samples,), cond_val, device=device)

        samples = model.sample(
            batch_size=n_samples,
            cond=cond,
            guidance_scale=guidance_scale,
            steps=steps,
            device=device,
        )

        gc_list = []
        motif_count = 0
        for i in range(n_samples):
            seq = decode_sequence(samples[i])
            gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
            gc_list.append(gc)
            if motif in seq:
                motif_count += 1
            if i < 4:
                sample_table.append({
                    'cond': label,
                    'seq': seq,
                    'gc': f'{gc:.1f}%',
                    'motif': motif in seq,
                })

        gc_mean = sum(gc_list) / len(gc_list)
        results[f'gen_gc_mean_{label}'] = gc_mean
        results[f'gen_gc_std_{label}'] = (
            sum((g - gc_mean) ** 2 for g in gc_list) / len(gc_list)
        ) ** 0.5
        results[f'gen_motif_rate_{label}'] = motif_count / n_samples

    gc_means = [results[f'gen_gc_mean_{l}'] for l in cond_labels]
    results['gen_gc_spread'] = max(gc_means) - min(gc_means)

    model.train()
    return results, sample_table


def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                    best_val_loss, config, path):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config,
    }, path)


def load_pretrained_weights(model, pretrain_path, logger):
    """Load pretrained backbone; condition modules are absent from the
    pretraining checkpoint and stay at their init (zero-init class emb +
    zero-init null_cond), so use strict=False and report the diff."""
    ckpt = torch.load(pretrain_path, map_location='cpu')
    pretrain_state = ckpt['model_state_dict']

    missing, unexpected = model.load_state_dict(pretrain_state, strict=False)

    logger.info(f"loaded pretrained weights: {pretrain_path}")
    logger.info(f"  matched: {len(pretrain_state) - len(unexpected)} params")
    if missing:
        logger.info(f"  new (init from scratch): {len(missing)}")
        for name in missing:
            logger.info(f"    + {name}")
    if unexpected:
        logger.info(f"  unused from ckpt: {len(unexpected)}")
        for name in unexpected:
            logger.info(f"    - {name}")

    return missing, unexpected
