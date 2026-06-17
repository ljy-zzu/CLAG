# CLAG: Closed-Loop LLM Augmentation for GoEmotions & SemEval

## Project layout

```
project/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/              # experiment hyperparameters
│   ├── semeval.yaml      
│   └── goemotions.yaml   
├── src/
│   ├── training/         # pipeline, agents, train scripts
│   ├── datasets/         
│   ├── models/           
│   └── utils/            # config loader, QC helpers
├── scripts/
│   ├── train.sh          
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
# copy 2018-E-c-En-{train,dev,test-gold}.json
# download roberta-large & Meta-Llama-3-8B-Instruct

# 3) Run SemEval experiment
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
4. **Training** — RoBERTa-large fine-tune (BCE or ASL) + early stop

## Config

All knobs live in YAML. Example (`configs/semeval.yaml`):

| Key | Value |
|-----|-------|
| `experiment.iter_train` | `asl` |
| `augmentation.gen_ratio` | `0.01` |
| `augmentation.qc_keep_ratio` | `0.60` |
| `training.epochs` | `5` |



## License

MIT — see [LICENSE](LICENSE).
