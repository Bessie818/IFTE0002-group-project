# IFTE0002 Group Project

Finance and Artificial Intelligence (2025/26)

## Project Overview

This project investigates credit card default prediction using two modelling pipelines:

- a transformer-based model built from scratch for tabular financial data
- a random forest benchmark used as a traditional machine learning baseline

The main objective is to evaluate whether a transformer architecture can achieve competitive performance on structured credit default data compared with a tuned random forest model.

## Key Contributions

- Implementation of a transformer model from scratch
- Tokenisation of tabular customer records into fixed-order token sequences
- Custom self-attention and multi-head attention modules
- Validation-based transformer training and model selection
- Comparison against random forest benchmark models
- Generation of result tables and figures for model evaluation

## Repository Structure

```text
src/
  transformer_tokenizer.py
  transformer_attention.py
  credit_card_transformer_model.py
  credit_card_transformer_model_training_d.py
  main.py
  main_training_d.py
  main_training_d_validated.py
  main_training_d_validated_layers.py
  main_training_d_validated_lr.py
  generate_transformer_figures.py

notebooks/
  random_forest_pipeline.ipynb

figures/
  transformer/
  rf/

results/
  model_comparison.csv
  classification_report.csv
  results_training_d.txt
  results_model_comparison.txt
  training_d_summary.md

models/
  best_rf_model.pkl
```

## How to Run

Run all commands from the project root directory.

### Transformer Experiments

Run the validated transformer model:

```bash
python src/main_training_d_validated.py
```

Run the layer-depth and attention-type ablation experiment:

```bash
python src/main_training_d_validated_layers.py
```

Generate transformer figures:

```bash
python src/generate_transformer_figures.py
```

### Random Forest Experiments

Open and run the random forest notebook:

```text
notebooks/random_forest_pipeline.ipynb
```

The trained random forest model is saved in:

```text
models/best_rf_model.pkl
```

## Experiments

### Transformer

The transformer experiments include:

- single-head vs multi-head attention
- learning rate tuning
- layer depth ablation
- validation-based model selection
- final test-set evaluation

The transformer uses a 64% training, 16% validation, and 20% testing split. The tokenizer is fitted only on the training set to avoid data leakage.

### Random Forest

The random forest benchmark includes:

- baseline random forest
- SMOTE-based random forest
- hyperparameter-tuned random forest
- feature importance analysis

## Outputs

Transformer figures are saved in:

```text
figures/transformer/
```

Random forest figures are saved in:

```text
figures/rf/
```

Experiment summaries and CSV outputs are saved in:

```text
results/
```

Saved models are stored in:

```text
models/
```

## Notes

The transformer model is designed to test whether sequence-based modelling can be applied to structured tabular credit data. The random forest benchmark remains a strong baseline for this task, while the transformer provides a competitive deep learning alternative.
