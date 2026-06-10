"""Sequence-level evaluation metrics for generated promoters."""

from collections import Counter

import numpy as np


def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq) * 100


def has_motif(seq, motif='GCGATCGC'):
    return motif in seq


def count_motif(seq, motif='GCGATCGC'):
    count = 0
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx == -1:
            break
        count += 1
        start = idx + 1
    return count


def edit_distance(s1, s2):
    """Hamming distance on equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def get_kmer_freq(sequences, k=4):
    counter = Counter()
    total = 0
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            counter[seq[i:i+k]] += 1
            total += 1
    return {kmer: count / total for kmer, count in counter.items()} if total > 0 else {}


def js_divergence(freq1, freq2):
    all_keys = set(freq1.keys()) | set(freq2.keys())
    p = np.array([freq1.get(k, 0) for k in all_keys])
    q = np.array([freq2.get(k, 0) for k in all_keys])

    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(jsd)


def positional_base_freq(sequences):
    """Per-position base frequencies. Returns [seq_len, 4] over ATGC."""
    seq_len = len(sequences[0])
    counts = np.zeros((seq_len, 4))
    base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    for seq in sequences:
        for i, base in enumerate(seq):
            if base in base_map:
                counts[i, base_map[base]] += 1
    return counts / len(sequences)


def evaluate_sequences(gen_seqs, real_seqs, train_seqs, log):
    """Full eval for a set of generated sequences. Returns metrics dict."""
    n = len(gen_seqs)
    metrics = {'n_samples': n}

    # GC
    gen_gc = [gc_content(s) for s in gen_seqs]
    real_gc = [gc_content(s) for s in real_seqs]
    metrics['gc_mean'] = float(np.mean(gen_gc))
    metrics['gc_std'] = float(np.std(gen_gc))
    metrics['gc_real_mean'] = float(np.mean(real_gc))
    metrics['gc_real_std'] = float(np.std(real_gc))
    log.info(f"  GC: gen {metrics['gc_mean']:.1f}±{metrics['gc_std']:.1f}% "
             f"vs real {metrics['gc_real_mean']:.1f}±{metrics['gc_real_std']:.1f}%")

    # GCGATCGC motif
    gen_motif_rate = sum(has_motif(s) for s in gen_seqs) / n
    real_motif_rate = sum(has_motif(s) for s in real_seqs) / len(real_seqs)
    metrics['motif_rate'] = float(gen_motif_rate)
    metrics['motif_rate_real'] = float(real_motif_rate)
    log.info(f"  motif GCGATCGC: gen {gen_motif_rate*100:.1f}% "
             f"vs real {real_motif_rate*100:.1f}%")

    # k-mer JSD
    for k in [3, 4, 5]:
        gen_kmer = get_kmer_freq(gen_seqs, k=k)
        real_kmer = get_kmer_freq(real_seqs, k=k)
        jsd = js_divergence(gen_kmer, real_kmer)
        metrics[f'kmer{k}_jsd'] = float(jsd)
    log.info(f"  k-mer JSD: 3-mer={metrics['kmer3_jsd']:.4f} "
             f"4-mer={metrics['kmer4_jsd']:.4f} "
             f"5-mer={metrics['kmer5_jsd']:.4f}")

    # diversity: pairwise Hamming on a random subsample
    n_pairs = min(500, n)
    indices = np.random.choice(n, size=n_pairs, replace=False)
    pair_dists = []
    for i in range(0, n_pairs - 1, 2):
        d = edit_distance(gen_seqs[indices[i]], gen_seqs[indices[i+1]])
        pair_dists.append(d)
    metrics['diversity_mean'] = float(np.mean(pair_dists))
    metrics['diversity_std'] = float(np.std(pair_dists))
    metrics['diversity_ratio'] = float(np.mean(pair_dists)) / len(gen_seqs[0])
    log.info(f"  diversity: Hamming {metrics['diversity_mean']:.1f}±{metrics['diversity_std']:.1f} "
             f"({metrics['diversity_ratio']*100:.1f}% of seq_len)")

    # novelty: min Hamming to a train subsample
    n_check = min(200, n)
    n_train_sample = min(2000, len(train_seqs))
    check_indices = np.random.choice(n, size=n_check, replace=False)
    train_sample = [train_seqs[i] for i in np.random.choice(len(train_seqs), size=n_train_sample, replace=False)]

    min_dists = []
    exact_match = 0
    for idx in check_indices:
        gen_s = gen_seqs[idx]
        min_d = min(edit_distance(gen_s, t) for t in train_sample)
        min_dists.append(min_d)
        if min_d == 0:
            exact_match += 1

    metrics['novelty_mean'] = float(np.mean(min_dists))
    metrics['novelty_std'] = float(np.std(min_dists))
    metrics['novelty_ratio'] = float(np.mean(min_dists)) / len(gen_seqs[0])
    metrics['exact_match_rate'] = exact_match / n_check
    log.info(f"  novelty: min Hamming {metrics['novelty_mean']:.1f}±{metrics['novelty_std']:.1f} "
             f"({metrics['novelty_ratio']*100:.1f}% of seq_len), "
             f"exact match {metrics['exact_match_rate']*100:.1f}%")

    # uniqueness
    unique_seqs = len(set(gen_seqs))
    metrics['uniqueness'] = unique_seqs / n
    log.info(f"  uniqueness: {unique_seqs}/{n} ({metrics['uniqueness']*100:.1f}%)")

    return metrics
