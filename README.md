# IFTE0002 Group Project

Group coursework for **IFTE0002 Finance and Artificial Intelligence 25/26**.

## Project topic
Credit card default prediction using:
- a transformer model built from scratch
- a random forest benchmark

## Current repository structure
- `src/main.py` — baseline training script
- `src/credit_card_transformer_model.py` — transformer classifier and transformer block
- `src/transformer_attention.py` — single-head and multi-head attention implementations
- `src/transformer_attention_self.py` — single-head self-attention reference file
- `src/transformer_tokenizer.py` — tokenizer for tabular credit-card records

## Notes
- The current codebase contains the baseline implementation shared within the group.
- Next steps include connecting multi-head attention into the full model, improving the training pipeline, and comparing single-head vs multi-head attention.
- Report files such as PDF and DOCX can be added later in separate folders if needed.

## Repository structure

- `src/` — transformer code and training scripts
- `notebooks/` — random forest benchmark notebook
- `figures/rf/` — EDA and random forest result figures
- `figures/transformer/` — reserved for transformer figures
- `results/` — experiment summaries and csv outputs
- `models/` — saved model artifacts