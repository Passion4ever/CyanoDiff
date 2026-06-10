#!/usr/bin/env python3
"""MDLM conditional generation + evaluation.

Loads a fine-tuned checkpoint, samples under each (guidance_scale, cond)
combination, and dumps metrics + per-combo CSVs.

Usage:
    python scripts/generate.py --checkpoint checkpoints/ft_v2_0330_152806/best_model.pt
    python scripts/generate.py --checkpoint ... --scales 0 1 2 3 5 --n-samples 1000
    python scripts/generate.py --checkpoint ... --steps 500 --batch-size 256

Output layout:
    results/{run_name}/
      generated_s{scale}_c{cond}.csv
      metrics.json
      eval.log
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.promoter_dataset import ExpressionBinner
from src.models.mdlm import MDLM
from src.utils import setup_logger
from src.metrics import gc_content, has_motif, evaluate_sequences
from src.sample import generate_sequences, generate_inpainting, parse_template


def main():
    parser = argparse.ArgumentParser(description='MDLM conditional generation + evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='fine-tuned checkpoint path')
    parser.add_argument('--scales', type=float, nargs='+', default=[0, 1, 2, 3, 5],
                        help='guidance scales')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='samples per (scale, cond) combo')
    parser.add_argument('--steps', type=int, default=1000,
                        help='sampling steps')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--binner', type=str, default=None,
                        help='ExpressionBinner .npy path')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='output dir; default runs/sample/{project}/{timestamp}/')
    parser.add_argument('--seed', type=int, default=42)

    # inpainting mode
    parser.add_argument('--inpaint', type=str, default=None,
                        help='template string (len=seq_len); ATGC=fixed, -/N=sample')
    parser.add_argument('--inpaint-cond', type=int, default=2,
                        help='inpaint cond class id (0=low, 1=mid, 2=high)')
    parser.add_argument('--inpaint-scale', type=float, default=1.0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ckpt_path = Path(args.checkpoint)
        project_name = ckpt_path.parent.parent.name
        timestamp = datetime.now().strftime('%m%d_%H%M%S')
        out_dir = Path('runs/sample') / project_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(out_dir, name='generate', log_file='eval.log')
    log.info(f"out dir: {out_dir}")
    log.info(f"checkpoint: {args.checkpoint}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"device: {device}")

    log.info("loading model")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt['config']
    n_cond_classes = config['model'].get('n_cond_classes', 0)
    discrete = n_cond_classes > 0

    model = MDLM(
        seq_len=config['data']['seq_len'],
        hidden_dim=config['model']['hidden_dim'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        ffn_dim=config['model']['ffn_dim'],
        dropout=0.0,
        has_expression_cond=True,
        n_cond_classes=n_cond_classes,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log.info(f"loaded (epoch {ckpt['epoch']+1}, val_loss={ckpt.get('best_val_loss', '?')})")
    log.info(f"cond mode: {'discrete (' + str(n_cond_classes) + ' classes)' if discrete else 'continuous'}")

    binner = None
    if discrete:
        binner_path = args.binner or (Path(args.checkpoint).parent / 'expression_binner.npy')
        if Path(binner_path).exists():
            binner = ExpressionBinner(n_bins=n_cond_classes).load(binner_path)
            stats = binner.get_bin_stats(
                pd.read_csv(config['data']['train_csv'])['log_expression'].values
            )
            for label, s in stats.items():
                log.info(f"  {label}: n={s['count']}  reads {s['reads_range']}")
        else:
            log.info(f"warn: binner not found at {binner_path}")

    log.info("loading real promoter data")

    # train set: used only for novelty comparison
    train_df = pd.read_csv(config['data']['train_csv'])
    train_seqs = train_df['sequence'].tolist()
    log.info(f"  train: {len(train_seqs)} (novelty reference)")

    # test set: used for quality comparison
    test_csv = config['data'].get('test_csv')
    if test_csv and Path(test_csv).exists():
        test_df = pd.read_csv(test_csv)
        test_seqs = test_df['sequence'].tolist()
        test_log_expr = test_df['log_expression'].values
        log.info(f"  test: {len(test_seqs)} (quality reference)")
    else:
        log.info("  warn: test_csv not found, using train for quality eval")
        test_df = train_df
        test_seqs = train_seqs
        test_log_expr = train_df['log_expression'].values

    # per-class test sequences for per-class quality comparison
    if discrete and binner is not None:
        test_bin_ids = binner.transform(test_log_expr)
        test_seqs_by_class = {}
        for c in range(n_cond_classes):
            mask = test_bin_ids == c
            test_seqs_by_class[c] = [test_seqs[i] for i in range(len(test_seqs)) if mask[i]]
            log.info(f"  class {c} test: {len(test_seqs_by_class[c])}")

    if args.inpaint:
        template = args.inpaint
        assert len(template) == config['data']['seq_len'], \
            f"template len {len(template)} != model seq_len {config['data']['seq_len']}"

        fixed_positions = parse_template(template)
        log.info("inpainting mode")
        log.info(f"  template: {template}")
        log.info(f"  fixed {len(fixed_positions)}/{len(template)} positions")
        log.info(f"  cond: class {args.inpaint_cond}, scale {args.inpaint_scale}")

        gen_seqs = generate_inpainting(
            model, template, fixed_positions,
            cond_value=args.inpaint_cond,
            guidance_scale=args.inpaint_scale,
            n_samples=args.n_samples,
            steps=args.steps,
            batch_size=args.batch_size,
            device=device,
            discrete=discrete,
        )

        # sanity check: do fixed positions survive the sample?
        n_correct = 0
        for seq in gen_seqs:
            ok = all(seq[p] == template[p].upper() for p in fixed_positions)
            if ok:
                n_correct += 1
        log.info(f"  fixed preservation: {n_correct}/{len(gen_seqs)} ({n_correct/len(gen_seqs)*100:.1f}%)")

        csv_path = out_dir / 'inpainted_sequences.csv'
        with open(csv_path, 'w') as f:
            f.write('sequence,gc_content,has_motif\n')
            for seq in gen_seqs:
                gc = gc_content(seq)
                motif = has_motif(seq)
                f.write(f'{seq},{gc:.2f},{motif}\n')
        log.info(f"  saved: {csv_path}")

        log.info(f"\n  samples (first 10):")
        log.info(f"  template: {template}")
        for seq in gen_seqs[:10]:
            # fixed positions in upper, sampled in lower
            display = ''.join(
                seq[i] if i in fixed_positions else seq[i].lower()
                for i in range(len(seq))
            )
            log.info(f"  gen: {display}  GC={gc_content(seq):.1f}%")

        log.info("inpainting done")
        return

    if discrete:
        cond_list = list(range(n_cond_classes))
        cond_labels = [binner.bin_labels[c] if binner else f'class{c}' for c in cond_list]
    else:
        cond_list = [0.1, 0.25, 0.4, 0.6, 0.8]
        cond_labels = [f'{c:.2f}' for c in cond_list]

    all_metrics = {}

    log.info("=" * 60)
    log.info(f"gen params: steps={args.steps}, n_samples={args.n_samples}")
    log.info(f"guidance scales: {args.scales}")
    log.info(f"conds: {cond_labels}")
    log.info(f"total to generate: {len(args.scales) * len(cond_list) * args.n_samples:,}")
    log.info("=" * 60)

    for scale in args.scales:
        log.info(f"\n{'='*60}")
        log.info(f"guidance scale = {scale}")
        log.info(f"{'='*60}")

        for cond_val, label in zip(cond_list, cond_labels):
            log.info(f"\n--- {label} (cond={cond_val}) ---")

            t0 = time.time()
            gen_seqs = generate_sequences(
                model, cond_val, scale, args.n_samples,
                args.steps, args.batch_size, device, discrete=discrete,
            )
            gen_time = time.time() - t0
            log.info(f"  generated {len(gen_seqs)} in {gen_time:.1f}s")

            # pick per-class test set for quality, full train set for novelty
            if discrete and binner is not None:
                real_for_compare = test_seqs_by_class[cond_val]
            else:
                real_for_compare = test_seqs

            metrics = evaluate_sequences(gen_seqs, real_for_compare, train_seqs, log)
            metrics['cond'] = cond_val
            metrics['cond_label'] = label
            metrics['guidance_scale'] = scale
            metrics['gen_time_sec'] = gen_time

            key = f's{scale}_{label}'
            all_metrics[key] = metrics

            csv_path = out_dir / f'generated_{key}.csv'
            with open(csv_path, 'w') as f:
                f.write('sequence,gc_content,has_motif\n')
                for seq in gen_seqs:
                    gc = gc_content(seq)
                    motif = has_motif(seq)
                    f.write(f'{seq},{gc:.2f},{motif}\n')

    log.info("\n" + "=" * 60)
    log.info("conditional control summary")
    log.info("=" * 60)

    for scale in args.scales:
        log.info(f"\n  guidance scale = {scale}:")
        gc_means = []
        motif_rates = []
        for label in cond_labels:
            key = f's{scale}_{label}'
            m = all_metrics[key]
            gc_means.append(m['gc_mean'])
            motif_rates.append(m['motif_rate'])
            log.info(f"    {label}: GC={m['gc_mean']:.1f}±{m['gc_std']:.1f}% "
                     f"motif={m['motif_rate']*100:.1f}% "
                     f"JSD4={m['kmer4_jsd']:.4f} "
                     f"div={m['diversity_ratio']*100:.1f}% "
                     f"novel={m['novelty_ratio']*100:.1f}% "
                     f"unique={m['uniqueness']*100:.1f}%")
        gc_spread = max(gc_means) - min(gc_means)
        log.info(f"    -> GC spread: {gc_spread:.2f}%")
        log.info(f"    -> motif range: {min(motif_rates)*100:.1f}% ~ {max(motif_rates)*100:.1f}%")

    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"\nmetrics saved: {metrics_path}")
    log.info(f"sequences saved: {out_dir}/generated_*.csv")
    log.info("done")


if __name__ == '__main__':
    main()
