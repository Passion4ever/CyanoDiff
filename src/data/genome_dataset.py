"""Pretraining datasets: whole-genome random crops and upstream-region crops."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from Bio import SeqIO


BASE_TO_ID = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
ID_TO_BASE = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
MASK_TOKEN_ID = 4
VOCAB_SIZE = 5


def encode_sequence(seq):
    return torch.tensor([BASE_TO_ID[b] for b in seq], dtype=torch.long)


def decode_sequence(ids):
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return ''.join(ID_TO_BASE.get(i, 'N') for i in ids)


def load_genomes(fasta_dir, val_ratio=0.05, seed=42):
    """Concatenate every contig per FASTA into one long string; split by genome."""
    fasta_dir = Path(fasta_dir)
    fasta_files = sorted(fasta_dir.glob('*.fna'))
    if not fasta_files:
        fasta_files = sorted(fasta_dir.glob('*.fasta'))
    if not fasta_files:
        raise FileNotFoundError(f"no .fna/.fasta under {fasta_dir}")

    print(f"found {len(fasta_files)} genome files")

    genomes = []
    total_bp = 0
    for f in fasta_files:
        seqs = [str(r.seq).upper() for r in SeqIO.parse(f, 'fasta')]
        if seqs:
            g = ''.join(seqs)
            genomes.append(g)
            total_bp += len(g)

    print(f"loaded {len(genomes)} genomes, {total_bp / 1e9:.2f} Gbp total")

    rng = random.Random(seed)
    indices = list(range(len(genomes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(genomes) * val_ratio))
    val_indices = set(indices[:n_val])

    train_genomes = [genomes[i] for i in range(len(genomes)) if i not in val_indices]
    val_genomes = [genomes[i] for i in range(len(genomes)) if i in val_indices]

    print(f"train: {len(train_genomes)} genomes, {sum(map(len, train_genomes)) / 1e9:.2f} Gbp")
    print(f"val:   {len(val_genomes)} genomes, {sum(map(len, val_genomes)) / 1e9:.2f} Gbp")

    return train_genomes, val_genomes


class GenomeDataset(Dataset):
    """On-the-fly random 81 bp crops from genomes, weighted by genome length.

    Segments containing non-ATGC bases (N, degenerate codes, etc.) are rejected
    and resampled.
    """

    def __init__(self, genomes, seq_len=81, samples_per_epoch=2_000_000):
        self.seq_len = seq_len
        self.samples_per_epoch = samples_per_epoch

        self.genomes = [g for g in genomes if len(g) >= seq_len]
        if len(self.genomes) < len(genomes):
            print(f"  dropped {len(genomes) - len(self.genomes)} genomes shorter than {seq_len}bp")

        lengths = [len(g) for g in self.genomes]
        total = sum(lengths)
        self.weights = [l / total for l in lengths]
        self.max_starts = [len(g) - seq_len for g in self.genomes]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        valid_bases = set('ATGC')
        for _ in range(100):
            genome_idx = random.choices(
                range(len(self.genomes)), weights=self.weights, k=1,
            )[0]
            genome = self.genomes[genome_idx]
            start = random.randint(0, self.max_starts[genome_idx])
            seq = genome[start:start + self.seq_len]
            if all(b in valid_bases for b in seq):
                return encode_sequence(seq)

        # Extremely rare fallback: 100 consecutive draws all contained N's.
        return torch.zeros(self.seq_len, dtype=torch.long)


def reverse_complement(seq):
    return seq[::-1].translate(str.maketrans('ATGC', 'TACG'))


def load_upstream_fasta(fasta_path, val_ratio=0.05, seed=42):
    """Load a FASTA of equal-length upstream windows; split at the sequence level."""
    fasta_path = Path(fasta_path)
    seqs = [
        str(r.seq).upper()
        for r in SeqIO.parse(fasta_path, 'fasta')
        if all(b in 'ATGC' for b in str(r.seq).upper())
    ]
    print(f"loaded {fasta_path}: {len(seqs)} seqs, length {len(seqs[0])}bp")

    rng = random.Random(seed)
    indices = list(range(len(seqs)))
    rng.shuffle(indices)
    n_val = max(1, int(len(seqs) * val_ratio))
    val_indices = set(indices[:n_val])

    train_seqs = [seqs[i] for i in range(len(seqs)) if i not in val_indices]
    val_seqs = [seqs[i] for i in range(len(seqs)) if i in val_indices]
    print(f"train: {len(train_seqs)}  val: {len(val_seqs)}")
    return train_seqs, val_seqs


class UpstreamDataset(Dataset):
    """Random 81 bp crops from fixed-length upstream windows + RC augmentation."""

    def __init__(self, seqs, seq_len=81, samples_per_epoch=2_000_000, rc_augment=True):
        self.seqs = seqs
        self.seq_len = seq_len
        self.samples_per_epoch = samples_per_epoch
        self.rc_augment = rc_augment
        self.max_offset = len(seqs[0]) - seq_len
        self.n_seqs = len(seqs)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seq = self.seqs[random.randint(0, self.n_seqs - 1)]
        offset = random.randint(0, self.max_offset)
        crop = seq[offset:offset + self.seq_len]
        if self.rc_augment and random.random() < 0.5:
            crop = reverse_complement(crop)
        return encode_sequence(crop)
