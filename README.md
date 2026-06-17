# CLAG: Closed-Loop LLM Augmentation for GoEmotions & SemEval

Iterative data augmentation with **Llama-3** generation, **RoBERTa-large** QC, and multi-label fine-tuning.

**Primary SemEval result:** `roberta-large-04-bce-plain-gen1pct`  
Baseline test macro-F1 **0.608** → after 6 rounds **0.617** (BCE, gen1pct, per-label QC top-60%).

## Project layout

```
project/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/              # experiment hyperparameters
│   ├── semeval.yaml      # ← main SemEval BCE gen1pct config
│   └── goemotions.yaml   # GoEmotions template
├── src/
│   ├── training/         # pipeline, agents, BERT train scripts
│   ├── datasets/         # SemEval labels & I/O
│   ├── models/           # semantic encoder
│   └── utils/            # config loader, QC helpers
├── scripts/
│   ├── train.sh          # baseline + 6-round pipeline
│   ├── evaluate.sh
│   └── inference.sh
├── data/                 # datasets (not in repo)
├── checkpoints/          # roberta-large, Llama (not in repo)
└── outputs/              # experiment runs (not in repo)
```

## Quick start

```bash
# 1) Install
conda env create -f environment.yml && conda activate emotion-clag
# or: pip install -r requirements.txt

# 2) Prepare data & models (see data/README.md)
mkdir -p data/semeval checkpoints
# copy 2018-E-c-En-{train,dev,test-gold}.json → data/semeval/
# download roberta-large & Meta-Llama-3-8B-Instruct → checkpoints/

# 3) Run SemEval BCE gen1pct experiment
bash scripts/train.sh

# 4) Evaluate best checkpoint
bash scripts/evaluate.sh
```

Override config:

```bash
CONFIG=configs/semeval.yaml bash scripts/train.sh -- --dry-run
```

## Method (4 modules)

1. **Analysis** — error-driven label quotas (~1% of train per round)
2. **Generation** — Llama-3 target-conditioned tweets
3. **Filtering** — fixed baseline classifier + semantic score, **top 60% per label**
4. **Training** — RoBERTa-large fine-tune (BCE or ASL) + dual eval macro-F1 + early stop

## Config

All knobs live in YAML. Example (`configs/semeval.yaml`):

| Key | Value |
|-----|-------|
| `experiment.iter_train` | `bce` |
| `augmentation.gen_ratio` | `0.01` |
| `augmentation.qc_keep_ratio` | `0.60` |
| `training.epochs` | `4` |
| `experiment.iterations` | `6` |

Switch to ASL: set `iter_train: asl_plain` (see `configs/goemotions.yaml` for GoEmotions defaults).

## Citation

Add your paper metadata here.

## License

MIT — see [LICENSE](LICENSE).
