#!/usr/bin/env python3
"""MDLM fine-tuning on 4-species promoters with a discrete expression condition.

A single shared config (configs/finetune.yaml) is used for all 4 species; the
species selects data paths and the run name via --species.

Usage:
    python scripts/finetune.py --config configs/finetune.yaml --species 7120 \\
        --pretrain runs/pretrain/pt_v3/<ts>/best_model.pt
    python scripts/finetune.py --resume runs/finetune/ft_v3_7120/<ts>/best_model.pt
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.promoter_dataset import PromoterDataset, ExpressionBinner
from src.models.mdlm import MDLM
from src.utils import (
    set_seed, get_cosine_schedule_with_warmup, count_parameters,
    create_run_dir, setup_logger, validate_conditional, evaluate_generation,
    save_checkpoint, load_pretrained_weights,
)


def main():
    parser = argparse.ArgumentParser(description='MDLM fine-tuning')
    parser.add_argument('--config', type=str, default='configs/finetune.yaml')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='pretraining checkpoint (overrides config)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from a fine-tuning checkpoint')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--species', type=str, default=None,
                        choices=['7120', '6803', 'MED4', 'MIT9313'],
                        help='target species; expands data CSV paths and run_name')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.no_wandb:
        config['output']['use_wandb'] = False
    if args.pretrain:
        config['pretrain']['checkpoint'] = args.pretrain

    if args.species:
        sp = args.species
        config['data']['train_csv'] = f'data/{sp}/processed_data/train.csv'
        config['data']['val_csv']   = f'data/{sp}/processed_data/val.csv'
        config['data']['test_csv']  = f'data/{sp}/processed_data/test.csv'
        config['output']['run_name'] = f'ft_v3_{sp}'

    set_seed(config['seed'])

    if args.resume:
        run_dir = Path(args.resume).parent
    else:
        run_dir = create_run_dir(
            config['output']['save_dir'],
            run_name=args.run_name or config['output'].get('run_name', 'finetune'),
        )

    shutil.copy2(args.config, run_dir / 'config.yaml')
    log = setup_logger(run_dir, name='finetune')
    log.info(f"run dir: {run_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    # data
    log.info("=" * 60)
    log.info("loading promoter data")

    n_cond_classes = config['model'].get('n_cond_classes', 0)

    if n_cond_classes > 0:
        import pandas as pd
        train_df = pd.read_csv(config['data']['train_csv'])
        binner = ExpressionBinner(n_bins=n_cond_classes)

        # Prefer GMM-precomputed edges when available; otherwise fit quantiles
        # from the training set.
        data_dir = Path(config['data']['train_csv']).parent
        precomputed_edges = data_dir / 'bin_edges.npy'
        if precomputed_edges.exists():
            binner.load(precomputed_edges)
            log.info(f"using precomputed bin edges: {precomputed_edges}")
        else:
            binner.fit(train_df['log_expression'].values)
            log.info("no precomputed bin_edges.npy; fitting quantiles on train")
        binner.save(run_dir / 'expression_binner.npy')

        stats = binner.get_bin_stats(train_df['log_expression'].values)
        log.info(f"discrete cond: {n_cond_classes} classes")
        for label, s in stats.items():
            log.info(f"  {label}: n={s['count']}  reads {s['reads_range']}")

        train_dataset = PromoterDataset(config['data']['train_csv'], binner=binner)
        val_dataset = PromoterDataset(config['data']['val_csv'], binner=binner)
    else:
        train_dataset = PromoterDataset(config['data']['train_csv'])
        val_dataset = PromoterDataset(config['data']['val_csv'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )

    # model
    log.info("=" * 60)
    log.info("building model")

    model = MDLM(
        seq_len=config['data']['seq_len'],
        hidden_dim=config['model']['hidden_dim'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        ffn_dim=config['model']['ffn_dim'],
        dropout=config['model']['dropout'],
        has_expression_cond=True,
        n_cond_classes=n_cond_classes,
    ).to(device)

    log.info(f"params: {count_parameters(model):,} ({count_parameters(model) / 1e6:.1f}M)")

    pretrain_ckpt = config['pretrain'].get('checkpoint')
    if pretrain_ckpt and not args.resume:
        load_pretrained_weights(model, pretrain_ckpt, log)
    elif not args.resume:
        log.info("no pretraining checkpoint — fine-tuning from scratch")

    # optimizer + schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['train']['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=config['train']['warmup_steps'],
        total_steps=total_steps,
        min_lr_ratio=config['train'].get('min_lr_ratio', 0.01),
    )

    log.info(f"batch size: {config['train']['batch_size']}")
    log.info(f"lr: {config['train']['lr']}")
    log.info(f"cfg_dropout: {config['train']['cfg_dropout']}")
    log.info(f"warmup: {config['train']['warmup_steps']}  steps/epoch: {steps_per_epoch}  total: {total_steps:,}")

    # resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        log.info(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        log.info(f"continuing at epoch {start_epoch}, global_step={global_step}")

    # wandb
    wandb_run = None
    if config['output'].get('use_wandb', False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['output']['wandb_project'],
                name=run_dir.name,
                config=config,
                dir=str(run_dir),
                resume='allow' if args.resume else None,
            )
            log.info(f"wandb: {wandb_run.url}")
        except Exception as e:
            log.info(f"wandb init failed: {e}")

    # Stage 1: freeze backbone, train condition modules only.
    # Without this warmup the zero-init class embeddings get whipped around by
    # the full LR and produce garbage during the first few hundred steps.
    cond_warmup_epochs = config['train'].get('cond_warmup_epochs', 0)

    if cond_warmup_epochs > 0 and not args.resume and n_cond_classes > 0:
        log.info("=" * 60)
        log.info(f"stage 1: freeze backbone, train condition modules ({cond_warmup_epochs} epochs)")

        cond_params = []
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'cond_emb.cond_emb_layer' in name or 'cond_emb.null_cond' in name:
                param.requires_grad = True
                cond_params.append(param)
                log.info(f"  trainable: {name}")
            else:
                param.requires_grad = False
                frozen_count += 1
        log.info(f"  frozen: {frozen_count} params")

        cond_optimizer = torch.optim.AdamW(cond_params, lr=1e-3, weight_decay=0.0)

        for warmup_epoch in range(cond_warmup_epochs):
            model.train()
            epoch_loss = 0.0
            for step, (seq, expr) in enumerate(train_loader):
                seq = seq.to(device)
                expr = expr.to(device)
                loss = model.compute_loss(
                    seq, cond=expr, cfg_dropout=config['train']['cfg_dropout'],
                )
                cond_optimizer.zero_grad()
                loss.backward()
                cond_optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            log.info(f"  [warmup {warmup_epoch+1}/{cond_warmup_epochs}] loss={avg_loss:.4f}")

        for param in model.parameters():
            param.requires_grad = True
        log.info("stage 1 done; unfroze everything")

        if hasattr(model.transformer.cond_emb, 'cond_emb_layer'):
            w = model.transformer.cond_emb.cond_emb_layer.weight.data
            norms = [f'{w[i].norm().item():.4f}' for i in range(w.shape[0])]
            log.info(f"  class embedding norm: {w.norm():.4f} (per-class: {norms})")

    # Stage 2: full fine-tuning.
    log.info("=" * 60)
    log.info("start fine-tuning" if cond_warmup_epochs == 0 else "stage 2: full fine-tuning")

    log_interval = config['output']['log_interval']
    val_interval = config['output']['val_interval']
    cfg_dropout = config['train']['cfg_dropout']

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, (seq, expr) in enumerate(train_loader):
            seq = seq.to(device)
            expr = expr.to(device)

            loss, metrics = model.compute_loss(
                seq, cond=expr, cfg_dropout=cfg_dropout, return_metrics=True,
            )
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config['train']['max_grad_norm'],
            ).item()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / (step + 1)
                acc = metrics.get('accuracy', 0) * 100
                ppl = metrics.get('perplexity', 0)
                log.info(
                    f"epoch {epoch+1}/{config['train']['epochs']} | "
                    f"step {step+1}/{steps_per_epoch} | "
                    f"loss {loss.item():.4f} | avg {avg_loss:.4f} | "
                    f"acc {acc:.1f}% | ppl {ppl:.3f} | "
                    f"lr {lr:.2e} | gn {grad_norm:.4f}"
                )
                if wandb_run:
                    import wandb
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': avg_loss,
                        'train/accuracy': metrics.get('accuracy', 0),
                        'train/perplexity': ppl,
                        'train/lr': lr,
                        'train/grad_norm': grad_norm,
                        'train/mask_ratio': metrics.get('mask_ratio', 0),
                        'global_step': global_step,
                    })

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, steps_per_epoch)
        log.info(f"[epoch {epoch+1}] train_loss={avg_epoch_loss:.4f}  time={epoch_time:.0f}s")

        if (epoch + 1) % val_interval == 0:
            val_metrics = validate_conditional(model, val_loader, device)
            val_loss = val_metrics['val_loss']
            val_acc = val_metrics['val_acc']

            log.info(f"[epoch {epoch+1}] val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")
            for bin_name in ['low', 'mid', 'high']:
                bl = val_metrics.get(f'val_loss_{bin_name}', -1)
                ba = val_metrics.get(f'val_acc_{bin_name}', -1)
                if bl >= 0:
                    log.info(f"  {bin_name}: loss={bl:.4f} acc={ba*100:.1f}%")

            base_accs = []
            for b in ['A', 'T', 'G', 'C']:
                ba = val_metrics.get(f'val_acc_{b}', -1)
                if ba >= 0:
                    base_accs.append(f'{b}={ba*100:.1f}%')
            if base_accs:
                log.info(f"  per-base: {' | '.join(base_accs)}")

            if wandb_run:
                import wandb
                wandb_dict = {'epoch': epoch + 1, 'global_step': global_step}
                for k, v in val_metrics.items():
                    wandb_dict[f'val/{k}'] = v
                wandb.log(wandb_dict)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                ckpt_path = run_dir / 'best_model.pt'
                save_checkpoint(model, optimizer, scheduler, epoch,
                                global_step, best_val_loss, config, ckpt_path)
                log.info(f"  saved best -> {ckpt_path.name} (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                log.info(f"  no improvement ({patience_counter}/{config['train']['early_stopping_patience']})")
                if patience_counter >= config['train']['early_stopping_patience']:
                    log.info(f"early stop: {patience_counter} epochs without improvement")
                    break

        gen_interval = config['output'].get('gen_interval', 20)
        if (epoch + 1) % gen_interval == 0:
            log.info(f"[epoch {epoch+1}] generation probe")
            gen_steps = config['output'].get('gen_eval_steps', 200)
            gen_metrics, sample_table = evaluate_generation(
                model, device, steps=gen_steps, n_samples=16,
                n_cond_classes=n_cond_classes,
            )
            if n_cond_classes > 0:
                cond_labels = [f'class{c}' for c in range(n_cond_classes)]
            else:
                cond_labels = ['c0.1', 'c0.3', 'c0.5', 'c0.7']
            for label in cond_labels:
                gc = gen_metrics.get(f'gen_gc_mean_{label}', 0)
                motif_r = gen_metrics.get(f'gen_motif_rate_{label}', 0)
                log.info(f"  {label}: GC={gc:.1f}% motif={motif_r*100:.0f}%")
            log.info(f"  GC spread (signal of cond control): {gen_metrics.get('gen_gc_spread', 0):.2f}%")

            for s in sample_table[:4]:
                log.info(f"  [{s['cond']}] {s['seq']} GC={s['gc']} motif={s['motif']}")

            if wandb_run:
                import wandb
                wandb_dict = {'epoch': epoch + 1, 'global_step': global_step}
                for k, v in gen_metrics.items():
                    wandb_dict[f'gen/{k}'] = v
                columns = ['cond', 'seq', 'gc', 'motif']
                table = wandb.Table(columns=columns)
                for s in sample_table:
                    table.add_data(s['cond'], s['seq'], s['gc'], s['motif'])
                wandb_dict['gen/samples'] = table
                wandb.log(wandb_dict)

        if (epoch + 1) % 50 == 0:
            ckpt_path = run_dir / f'checkpoint_epoch{epoch+1}.pt'
            save_checkpoint(model, optimizer, scheduler, epoch,
                            global_step, best_val_loss, config, ckpt_path)
            log.info(f"  saved checkpoint -> {ckpt_path.name}")

    # done
    final_path = run_dir / 'final_model.pt'
    save_checkpoint(model, optimizer, scheduler, epoch,
                    global_step, best_val_loss, config, final_path)

    log.info("=" * 60)
    log.info("fine-tuning done")
    log.info(f"best val_loss: {best_val_loss:.4f}")
    log.info(f"run dir: {run_dir}")

    log.info("final generation probe (full steps):")
    gen_metrics, sample_table = evaluate_generation(
        model, device, steps=config['diffusion']['steps'], n_samples=32,
        n_cond_classes=n_cond_classes,
    )
    if n_cond_classes > 0:
        cond_labels = [f'class{c}' for c in range(n_cond_classes)]
    else:
        cond_labels = ['c0.1', 'c0.3', 'c0.5', 'c0.7']
    for label in cond_labels:
        gc = gen_metrics.get(f'gen_gc_mean_{label}', 0)
        gc_std = gen_metrics.get(f'gen_gc_std_{label}', 0)
        motif_r = gen_metrics.get(f'gen_motif_rate_{label}', 0)
        log.info(f"  {label}: GC={gc:.1f}±{gc_std:.1f}% motif={motif_r*100:.0f}%")

    log.info(f"  GC spread: {gen_metrics.get('gen_gc_spread', 0):.2f}%")
    log.info("samples:")
    for s in sample_table:
        log.info(f"  [{s['cond']}] {s['seq']} GC={s['gc']} motif={s['motif']}")

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
