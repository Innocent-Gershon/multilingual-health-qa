# Multilingual Health QA — Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innocent-Gershon/multilingual-health-qa/blob/main/notebooks/02_Training_and_Inference.ipynb)

**Zindi Competition:** Multilingual Health Question Answering in Low-Resource African Languages  
**Task:** Generate health answers in Akan, Amharic, Luganda, Swahili, and English  
**Model:** google/mt5-small fine-tuned with LoRA (PEFT)  
**Metrics:** ROUGE-1 F1, ROUGE-L F1, LLM-as-a-Judge

---

## Dataset

| Split | Rows | Subsets |
|-------|------|---------|
| Train | 29,815 | Aka_Gha, Amh_Eth, Lug_Uga, Swa_Ken, Eng_Uga, Eng_Gha, Eng_Eth, Eng_Ken |
| Val   | 6,686  | Same 8 subsets |
| Test  | 2,618  | Same 8 subsets (no answers) |

---

## Project Structure

```
multilingual-health-qa/
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis
│   └── 02_Training_and_Inference.ipynb  # Full training + submission pipeline
├── src/
│   ├── preprocess.py                 # Data loading and prompt construction
│   ├── train.py                      # Training script (standalone)
│   ├── inference.py                  # Inference + submission generation
│   └── evaluate.py                   # ROUGE scoring utilities
├── configs/
│   └── config.yaml                   # All hyperparameters
├── submissions/                      # Generated submission CSV files
├── requirements.txt
└── README.md
```

---

## Quick Start on Google Colab

1. Open the training notebook via the Colab badge above
2. Run **Step 1** to install dependencies
3. Run **Step 2** to upload `Train.csv`, `Val.csv`, `Test.csv`
4. Run all remaining steps sequentially
5. Download `submission.csv` at the final step and submit to Zindi

---

## Local Setup

```bash
git clone https://github.com/Innocent-Gershon/multilingual-health-qa.git
cd multilingual-health-qa
pip install -r requirements.txt

# Place Train.csv, Val.csv, Test.csv in the root directory
cd src
python train.py
python inference.py
```

---

## Experiments Tracked

| # | Change | ROUGE-L |
|---|--------|---------|
| 1 | Baseline mt5-small, no LoRA, 1 epoch | TBD |
| 2 | + LoRA (r=16) | TBD |
| 3 | + Prompt engineering (language prefix) | TBD |
| 4 | Increase epochs to 3 | TBD |
| 5 | max_target_length 128 → 256 | TBD |
| 6 | lr 5e-4 → 3e-4 | TBD |
| 7 | num_beams 1 → 4 | TBD |
| 8 | LoRA r=8 → r=16 | TBD |
| 9 | mt5-small → mt5-base | TBD |
| 10 | no_repeat_ngram_size=3 | TBD |

---

## Reproducibility

All seeds are fixed at `42`. To reproduce exactly:
```python
CFG['seed'] = 42
```
Results may vary slightly across different GPU hardware.

---

## Ethical Considerations

This system generates health information in low-resource languages. Generated answers should not replace professional medical advice. Models may reflect biases present in training data.
