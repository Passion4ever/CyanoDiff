#!/usr/bin/env python3
"""MDLM pretraining on cyanobacterial genomes.

Usage:
    python scripts/pretrain.py --config configs/pretrain.yaml
    python scripts/pretrain.py --config configs/pretrain.yaml --no-wandb
    # LTSO: override fasta dir + run name from a single shared config
    python scripts/pretrain.py --config configs/pretrain.yaml \\
        --fasta-dir data/cyanobacteria_genomes/fasta_ltso_6803 --run-name pt_ltso_6803
    python scripts/pretrain.py --resume runs/pretrain/pt_v1/0309_153000/best_model.pt
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

from src.data.genome_dataset import (
    GenomeDataset, load_genomes, decode_sequence,
    UpstreamDataset, load_upstream_fasta,
)
from src.models.mdlm import MDLM
from src.utils import (
    set_seed, get_cosine_schedule_with_warmup, count_parameters,
    create_run_dir, setup_logger, validate, save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description='MDLM pretraining')
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from checkpoint (.pt); reuses its run dir')
    parser.add_argument('--init-from', type=str, default=None,
                        help='load model weights only; fresh optimizer/scheduler/epoch in a new run dir')
    parser.add_argument('--run-name', type=str, default=None,
                        help='overrides output.run_name in the config')
    parser.add_argument('--fasta-dir', type=str, default=None,
                        help='overrides data.fasta_dir (LTSO: use _ltso_<species> subset)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.no_wandb:
        config['output']['use_wandb'] = False

    if args.fasta_dir:
        config['data']['fasta_dir'] = args.fasta_dir

    set_seed(config['seed'])

    if args.resume:
        run_dir = Path(args.resume).parent
    else:
        run_dir = create_run_dir(
            config['output']['save_dir'],
            run_name=args.run_name or config['output'].get('run_name', 'pretrain'),
        )

    shutil.copy2(args.config, run_dir / 'config.yaml')

    log = setup_logger(run_dir, name='pretrain')
    log.info(f"run dir: {run_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    # data
    log.info("=" * 60)

    upstream_fasta = config['data'].get('upstream_fasta')

    if upstream_fasta:
        log.info("loading upstream-region dataset")
        train_seqs, val_seqs = load_upstream_fasta(
            upstream_fasta,
            val_ratio=config['data']['val_ratio'],
            seed=config['seed'],
        )
        train_dataset = UpstreamDataset(
            train_seqs,
            seq_len=config['data']['seq_len'],
            samples_per_epoch=config['data']['samples_per_epoch'],
            rc_augment=config['data'].get('rc_augment', True),
        )
        val_dataset = UpstreamDataset(
            val_seqs,
            seq_len=config['data']['seq_len'],
            samples_per_epoch=config['data']['val_samples'],
            rc_augment=False,  # no augmentation at val
        )
    else:
        log.info("loading whole-genome dataset")
        train_genomes, val_genomes = load_genomes(
            config['data']['fasta_dir'],
            val_ratio=config['data']['val_ratio'],
            seed=config['seed'],
        )
        train_dataset = GenomeDataset(
            train_genomes,
            seq_len=config['data']['seq_len'],
            samples_per_epoch=config['data']['samples_per_epoch'],
        )
        val_dataset = GenomeDataset(
            val_genomes,
            seq_len=config['data']['seq_len'],
            samples_per_epoch=config['data']['val_samples'],
        )

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
        has_expression_cond=False,  # pretraining: unconditional
    ).to(device)

    n_params = count_parameters(model)
    log.info(f"params: {n_params:,} ({n_params / 1e6:.1f}M)")

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
    )

    log.info(f"batch size: {config['train']['batch_size']}")
    log.info(f"lr: {config['train']['lr']}")
    log.info(f"warmup steps: {config['train']['warmup_steps']}")
    log.info(f"steps/epoch: {steps_per_epoch:,}")
    log.info(f"total steps: {total_steps:,}")
    log.info(f"schedule: cosine w/ warmup")

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

    if args.init_from:
        log.info(f"init-from (weights only): {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        prev_loss = ckpt.get('best_val_loss', '?')
        log.info(f"  prior best_val_loss={prev_loss}; training restarts at epoch 0")

    # wandb
    wandb_run = None
    if config['output'].get('use_wandb', True):
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
            log.info(f"wandb init failed: {e}; continuing without")

    # training loop
    log.info("=" * 60)
    log.info("starting training")

    log_interval = config['output']['log_interval']
    val_interval = config['output']['val_interval']

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            x_0 = batch.to(device)
            loss, metrics = model.compute_loss(x_0, return_metrics=True)

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
                    f"loss {loss.item():.4f} | "
                    f"avg {avg_loss:.4f} | "
                    f"acc {acc:.1f}% | "
                    f"ppl {ppl:.3f} | "
                    f"lr {lr:.2e} | "
                    f"gn {grad_norm:.4f}"
                )

                if wandb_run:
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
        avg_epoch_loss = epoch_loss / steps_per_epoch
        log.info(f"[epoch {epoch+1}] train_loss={avg_epoch_loss:.4f}  time={epoch_time:.0f}s")

        if (epoch + 1) % val_interval == 0:
            val_loss, val_acc = validate(model, val_loader, device)
            log.info(f"[epoch {epoch+1}] val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")

            if wandb_run:
                wandb.log({
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'epoch': epoch + 1,
                    'global_step': global_step,
                })

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

        if (epoch + 1) % 10 == 0:
            ckpt_path = run_dir / f'checkpoint_epoch{epoch+1}.pt'
            save_checkpoint(model, optimizer, scheduler, epoch,
                            global_step, best_val_loss, config, ckpt_path)
            log.info(f"  saved checkpoint -> {ckpt_path.name}")

    # done
    final_path = run_dir / 'final_model.pt'
    save_checkpoint(model, optimizer, scheduler, epoch,
                    global_step, best_val_loss, config, final_path)

    log.info("=" * 60)
    log.info("pretraining done")
    log.info(f"best val_loss: {best_val_loss:.4f}")
    log.info(f"run dir: {run_dir}")

    log.info("sample 5 sequences:")
    model.eval()
    samples = model.sample(batch_size=5, steps=config['diffusion']['steps'], device=device)
    for i, s in enumerate(samples):
        seq = decode_sequence(s)
        gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
        log.info(f"  [{i+1}] {seq} (GC={gc:.1f}%)")

    if wandb_run:
        wandb.finish()


if __name__ == '__main__':
    main()
