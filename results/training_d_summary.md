# Training (D) Summary

## 1. My role

I was mainly responsible for the Transformer training part of the group project.

My work included:
- connecting tokenized tabular inputs to the Transformer model,
- running and improving the training pipeline,
- comparing single-head and multi-head attention,
- tuning training-related hyperparameters,
- improving the evaluation protocol using a validated pipeline,
- handling class imbalance with weighted loss,
- generating figures and result summaries for the final report.

---

## 2. Development process

The Transformer work was completed in two stages.

### Early exploratory stage
In the earlier stage, I ran exploratory experiments on:
- single-head vs multi-head attention,
- number of layers,
- dropout settings,
- initial validation-based runs.

These experiments were useful for understanding the model behaviour and identifying promising settings, but they are not the final official results used in the report.

### Final rebuilt stage
To make the Transformer results more rigorous and aligned with the final project requirements, I rebuilt the training workflow using:
- a proper train / validation / test split,
- tokenizer fitted on the training set only,
- validation-based model selection,
- early stopping,
- weighted BCEWithLogitsLoss for class imbalance,
- staged hyperparameter tuning.

The final official Transformer results are based on this rebuilt training pipeline.

---

## 3. Final official training pipeline

The final Transformer training pipeline used:
- train / validation / test split,
- tokenizer fitted on the training set only,
- weighted BCEWithLogitsLoss,
- AdamW optimizer,
- early stopping based on validation AUC,
- final test evaluation only once after model selection.

Dataset split:
- Train size: 19200
- Validation size: 4800
- Test size: 6000

---

## 4. Class imbalance handling

The final model used weighted BCEWithLogitsLoss:

`criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)`

This was used because the credit default dataset is imbalanced, where the default class is the minority class.

Note:
- the weighted loss changes the numerical scale of the loss values,
- therefore the final loss values should not be directly compared with older unweighted runs.

---

## 5. Final staged hyperparameter tuning

A staged tuning process was performed.

### Stage 1: learning rate
Tested:
- 1e-3
- 5e-4
- 1e-4

Best result:
- best learning rate = `0.001`

### Stage 2: d_model
Tested:
- 16
- 32
- 64

Best result:
- best d_model = `16`

### Stage 3: dropout
Tested:
- 0.1
- 0.2
- 0.3

Best result:
- best dropout = `0.3`

### Stage 4: attention type and layer depth
Tested:
- single-head, layers = 1 / 2 / 3
- multi-head, layers = 1 / 2 / 3

Best result:
- best attention setting = `multi-head, 3 layers`

### Stage 5: batch size
Tested:
- 32
- 64
- 128

Best result:
- best batch size = `32`

---

## 6. Final selected Transformer configuration

The final best Transformer configuration selected using validation only was:

- attention_type = `multi`
- num_layers = `3`
- dropout = `0.3`
- learning_rate = `0.001`
- d_model = `16`
- attention_dim = `16`
- ffn_hidden_dim = `32`
- batch_size = `32`
- max_epochs = `30`
- patience = `5`

Best validation result:
- Best Epoch = `16`
- Best Validation AUC = `0.7745`

Final test result:
- Final Test Loss = `0.8890`
- Final Test Accuracy = `0.7518`
- Final Test AUC = `0.7735`

---

## 7. Main interpretation

The final rebuilt experiments showed that:

1. a validated pipeline was essential for proper model selection;
2. weighted BCEWithLogitsLoss was a suitable improvement for the imbalanced default prediction task;
3. smaller model size worked better in this tabular classification setting, with `d_model = 16` outperforming larger dimensions;
4. stronger regularization helped under the final setup, with `dropout = 0.3` performing best;
5. the final best Transformer was a `multi-head, 3-layer` model;
6. the final test AUC of `0.7735` shows that the Transformer is competitive on this task.

---

## 8. Overfitting discussion

In earlier exploratory experiments, some deeper single-head settings showed mild signs of overfitting, where train loss decreased but validation improvement was limited.

However, in the final rebuilt pipeline, the use of:
- weighted loss,
- early stopping,
- a validated split,
- and more systematic tuning

helped reduce this issue.

Under the final setup, the best model was a 3-layer multi-head Transformer, which indicates that deeper models can still work well when the training setup is properly controlled.

---

## 9. Comparison with Random Forest

The tuned Random Forest benchmark achieved approximately:

- Test AUC = `0.777`

The final rebuilt Transformer achieved:

- Test AUC = `0.7735`

This shows that the final Transformer was very close to the tuned Random Forest benchmark, although Random Forest still performed slightly better overall.

---

## 10. Final reporting rule

For the final group report, all official Transformer results should use the rebuilt final pipeline results from:

- `src/main_training_d_full_rebuild.py`

and the generated files:

- `results/transformer_full_tuning_results.csv`
- `results/transformer_best_run_curves.csv`
- `results/transformer_best_config.txt`

Earlier results should only be referenced as exploratory development steps, not as the final reported benchmark.

---

## 11. Final figures to update later

The final Transformer figures should be updated based on the rebuilt pipeline results.

Recommended final figure set:
- validation AUC curve of the final best run
- train loss curve of the final best run
- learning rate tuning figure
- d_model tuning figure
- dropout tuning figure
- attention/layer comparison figure
- final Transformer vs Random Forest comparison figure

The earlier exploratory comparison figure can be kept only as an appendix figure.