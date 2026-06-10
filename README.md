# CyanoDiff

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20590300.svg)](https://doi.org/10.5281/zenodo.20590300)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official repository for *"CyanoDiff: Class-Conditional Cyanobacterial Promoter
Generation via Masked Diffusion Language Modeling"*.

CyanoDiff is a masked diffusion language model for designing cyanobacterial
promoters with controllable expression strength. Pretrained on cyanobacterial
genomes and fine-tuned on labeled promoters, it generates 81 bp promoters
steered toward a low / mid / high expression class via classifier-free
guidance.


## Install

```bash
git clone https://github.com/Passion4ever/CyanoDiff.git
cd CyanoDiff
conda create -n prom python=3.10 -y
conda activate prom
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Usage

**Pretrain** on genomes:

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

**Fine-tune** on one of the four built-in species (`--species` selects
7120 / 6803 / MED4 / MIT9313 and expands its data paths):

```bash
python scripts/finetune.py --config configs/finetune.yaml \
    --species 7120 --pretrain path/to/pretrained.pt
```

To fine-tune on **your own species**, prepare `train.csv` / `val.csv` /
`test.csv`, each with two columns:

- `sequence` — an 81 bp promoter (`A/T/G/C`), 60 bp upstream + TSS + 20 bp down
- `log_expression` — `log10(reads + 1)` of that promoter's expression

Point the config at them and run without `--species` (expression is binned into
low / mid / high quantile classes automatically from your training set):

```bash
python scripts/finetune.py --config configs/finetune.yaml \
    --pretrain path/to/pretrained.pt --run-name my_species \
    # set data.{train,val,test}_csv in configs/finetune.yaml to your CSVs
```

**Generate** sequences per expression class across guidance scales:

```bash
python scripts/generate.py --checkpoint path/to/finetuned.pt \
    --scales 1 --n-samples 1000
```

This writes per-class CSVs and a `metrics.json` (GC content, GCGATCGC motif
rate, k-mer JSD, diversity, novelty, uniqueness).

## Data & checkpoints

Genomes, processed promoter datasets, and trained checkpoints are **not** in
this repository — they are hosted on Zenodo:

- **Zenodo**: https://doi.org/10.5281/zenodo.20590300

The deposit bundles `data.tar.gz` (486 genomes + four-species promoter
datasets and their source tables), `checkpoints.tar.gz` (pretrained model +
four fine-tuned models), and `code.tar.gz` (this model-core package). Download
and extract into the repository root:

```bash
tar -xzf data.tar.gz
tar -xzf checkpoints.tar.gz
```

## Citation

If you use CyanoDiff, please cite the dataset deposit (a paper citation will be
added upon publication):

```bibtex
@dataset{cyanodiff_2026,
  author    = {Yang, Guang and Li, Jianing and Kwoh, Chee Keong and
               Hu, Jinlu and Shi, Jian-Yu},
  title     = {{CyanoDiff: data, model checkpoints, and code for
               class-conditional cyanobacterial promoter generation via
               masked diffusion language modeling}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20590300},
  url       = {https://doi.org/10.5281/zenodo.20590300}
}
```

## License

See [LICENSE](LICENSE).
