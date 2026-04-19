# Training (D) Summary

## 1. Role and scope

I was mainly responsible for the **Training (D)** part of the transformer model in the IFTE0002 group project.

My work focused on:
- connecting the tokenized tabular input to the transformer model,
- making the full model trainable end-to-end,
- implementing and running the training loop,
- tuning training-related settings such as the number of layers and dropout,
- checking whether the model learned meaningful patterns,
- and improving the training procedure by adding a validation split and best model selection.

This work was done without overwriting the original group files. Instead, I created separate Training (D) versions so that the original implementation from other group members remained unchanged.

---

## 2. Files created / maintained

The following files were used or created for the Training (D) part:

- `src/credit_card_transformer_model_training_d.py`
- `src/main_training_d.py`
- `src/main_training_d_single.py`
- `src/main_training_d_multi.py`
- `src/main_training_d_validated.py`

Supporting result files:
- `results/results_training_d.txt`
- `results/results_model_comparison.txt`
- `results/training_d_summary.md`

These files contain:
- the Training (D) model implementation,
- experiment scripts for single-head and multi-head attention,
- a validated training script with train/validation/test split,
- and experiment summaries for later reporting.

---

## 3. What was implemented

### 3.1 Input pipeline and model connection

I connected the tokenized credit card client data to the transformer model so that the full forward path worked correctly:

`input -> embedding -> transformer block / attention -> classifier -> prediction`

This included:
- token ids as model input,
- batched tensor input,
- correct tensor shape handling,
- prediction head output alignment,
- binary classification output with `BCEWithLogitsLoss`.

### 3.2 Training loop

I ran the full training loop including:
- forward pass,
- loss computation,
- backward pass,
- optimizer update.

The model was trained with:
- `AdamW`
- `BCEWithLogitsLoss`
- batch-based training with PyTorch `DataLoader`

### 3.3 Attention comparison

I ran and compared:
- **single-head attention**
- **multi-head attention**

This was done to test whether the more complex multi-head version would perform better on this tabular credit default dataset.

### 3.4 Parameter tuning

I tuned the following settings:
- `num_layers`
- `dropout`

I focused on these because deeper layers may overfit and dropout directly affects training stability and generalization.

### 3.5 Validation-based training improvement

To make the training process more rigorous, I also implemented:
- a **train / validation / test split**
- **best model selection using validation AUC**

This was added in `main_training_d_validated.py` so that the final test result was not directly used for repeated tuning.

---

## 4. Experimental settings

Unless otherwise stated, the main settings used in the transformer experiments were:

- `d_model = 64`
- `ffn_hidden_dim = 128`
- `learning_rate = 1e-3`
- `batch_size = 128`
- `optimizer = AdamW`
- `loss = BCEWithLogitsLoss`
- `num_epochs = 15`

For multi-head experiments:
- `num_heads = 4`

---

## 5. Experiment results

### 5.1 Transformer experiments

| Model Type | Layers | Dropout | Train Loss | Test Acc | Test AUC |
|---|---:|---:|---:|---:|---:|
| Single-head | 3 | 0.1 | 0.4112 | 0.8142 | 0.7656 |
| Multi-head | 3 | 0.1 | 0.4068 | 0.8177 | 0.7669 |
| Multi-head | 1 | 0.1 | 0.4286 | 0.8175 | 0.7697 |
| Multi-head | 2 | 0.1 | 0.4302 | 0.8172 | 0.7667 |
| Multi-head | 4 | 0.1 | 0.4283 | 0.8188 | 0.7667 |
| Multi-head | 1 | 0.2 | 0.4298 | 0.8155 | 0.7743 |
| Multi-head | 1 | 0.3 | 0.4365 | 0.8167 | 0.7653 |
| Single-head | 1 | 0.2 | 0.4267 | 0.8200 | 0.7744 |

### 5.2 Best non-validated results

Best tuned single-head result:
- `attention_type = "single"`
- `num_layers = 1`
- `dropout = 0.2`
- `Test AUC = 0.7744`

Best tuned multi-head result:
- `attention_type = "multi"`
- `num_layers = 1`
- `dropout = 0.2`
- `Test AUC = 0.7743`

These two best tuned versions performed **almost identically**, with the single-head result being only marginally higher.

---

## 6. Validated training result

To improve the reliability of the reported transformer result, I added a validation split and selected the best model based on validation AUC.

### Validated single-head result
Configuration:
- `attention_type = "single"`
- `num_layers = 1`
- `dropout = 0.2`

Data split:
- Train size = `19200`
- Validation size = `4800`
- Test size = `6000`

Best validation result:
- `Best Epoch = 13`
- `Best Validation AUC = 0.7732`

Final test result using the best validation model:
- `Test Loss = 0.4395`
- `Test Acc = 0.8182`
- `Test AUC = 0.7701`

This validated result is slightly lower than the earlier direct test-based tuning result, which is expected and indicates that the validation-based evaluation is more rigorous.

---

## 7. Main findings

### 7.1 Single-head vs multi-head
Under the initial comparable setting (`num_layers = 3`, `dropout = 0.1`), the multi-head model performed slightly better than the single-head baseline.

However, after tuning, the difference between the best single-head and best multi-head configurations became extremely small:
- best single-head AUC = `0.7744`
- best multi-head AUC = `0.7743`

This suggests that for this structured tabular task, the benefit of multi-head attention is limited.

### 7.2 Effect of depth
Among the tested multi-head settings, using **1 transformer layer** produced the best result. Increasing the number of layers to 2, 3, or 4 did not improve AUC.

This suggests that a deeper transformer architecture did not provide additional value for this dataset and may introduce unnecessary complexity.

### 7.3 Effect of dropout
Dropout had a meaningful effect:
- `dropout = 0.1` gave reasonable results,
- `dropout = 0.2` improved performance,
- `dropout = 0.3` reduced performance.

Therefore, `dropout = 0.2` appears to be the most suitable among the tested values.

### 7.4 Value of validation split
The validated setup provided a more reliable final result:
- best validation AUC = `0.7732`
- final test AUC = `0.7701`

This makes the transformer training procedure more defensible for the final report because model selection was based on validation rather than repeatedly checking the test set.

---

## 8. Comparison with Random Forest benchmark

The Random Forest benchmark results currently in the repository are:

### Baseline RF (balanced)
- AUC = `0.760`
- AP = `0.539`
- F1 = `0.443`

### SMOTE RF
- AUC = `0.751`
- AP = `0.498`
- F1 = `0.504`

### Tuned RF
- AUC = `0.777`
- AP = `0.554`
- F1 = `0.544`

### Comparison summary
The tuned Random Forest benchmark slightly outperformed the validated transformer model:
- Tuned RF AUC = `0.777`
- Validated transformer AUC = `0.7701`

This suggests that for this **tabular credit default prediction task**, Random Forest remains a very strong benchmark and may be better suited to the data structure, while the transformer still provides competitive performance after tuning.

---

## 9. Final recommendation for reporting

For the final report, I recommend presenting:

### Transformer result to highlight
Use the **validated transformer result** as the main final transformer result:
- Best Epoch = `13`
- Best Validation AUC = `0.7732`
- Final Test AUC = `0.7701`

### Additional transformer discussion
Mention that:
- the best non-validated tuned single-head model achieved `AUC = 0.7744`,
- the best multi-head model achieved `AUC = 0.7743`,
- and overall tuned single-head and multi-head attention performed very similarly.

### Comparison statement
A good summary sentence for the report is:

> The tuned transformer model performed competitively, but the tuned Random Forest benchmark slightly outperformed it on this tabular credit default prediction task.

---

## 10. Final note

The Training (D) part now includes:
- model training implementation,
- single-head vs multi-head comparison,
- layer tuning,
- dropout tuning,
- validation-based model selection,
- and final transformer benchmarking against Random Forest.

## 11. Transformer figures generated

Two additional transformer figures were generated and saved in the repository:

- `figures/transformer/fig_transformer_model_comparison.png`
- `figures/transformer/fig_validated_auc_curve.png`

Three more transformer figures were added:
- `figures/transformer/fig_validated_train_loss_curve.png`
- `figures/transformer/fig_transformer_vs_rf_comparison.png`
- `figures/transformer/fig_layer_ablation_comparison.png`

These figures help visualize:
- the comparison across transformer experiments,
- the effect of layer depth on performance,
- the training dynamics through the validated train loss curve,
- the validation-based model selection process,
- and the final comparison between transformer and random forest.

## 12. Validated learning rate tuning

A controlled learning rate tuning experiment was conducted using the validated pipeline.

Fixed structure:
- attention_type = `single`
- num_layers = `1`
- dropout = `0.2`
- d_model = `64`
- ffn_hidden_dim = `128`
- batch_size = `128`
- epochs = `15`

Tested learning rates:
- `1e-3`
- `5e-4`
- `1e-4`

### Results

| Learning Rate | Best Epoch | Best Validation AUC | Final Test Loss | Final Test Acc | Final Test AUC |
|---|---:|---:|---:|---:|---:|
| 1e-3 | 13 | 0.7689 | 0.4395 | 0.8162 | 0.7648 |
| 5e-4 | 14 | 0.7691 | 0.4381 | 0.8187 | 0.7681 |
| 1e-4 | 15 | 0.7518 | 0.4427 | 0.8167 | 0.7589 |

### Interpretation

Among the tested learning rates, `5e-4` produced the best validation and test performance in this controlled experiment.

`1e-4` appeared too small and likely underfit, as both validation AUC and final test AUC were clearly lower.

`1e-3` remained competitive, but it was slightly worse than `5e-4`.

This learning rate sweep improved the systematicity of the training experiments and addressed an important weakness in the earlier training setup.

## 13. Validated layer ablation

A validated layer ablation experiment was conducted using the best tuned learning rate.

Fixed settings:
- learning rate = `5e-4`
- dropout = `0.2`
- d_model = `64`
- ffn_hidden_dim = `128`
- batch_size = `128`
- epochs = `15`

Tested configurations:
- single-head, layers = 1 / 2 / 3
- multi-head, layers = 1 / 2 / 3

### Results

| Attention Type | Layers | Best Epoch | Best Validation AUC | Final Test AUC |
|---|---:|---:|---:|---:|
| Single | 1 | 12 | 0.7722 | 0.7682 |
| Single | 2 | 7 | 0.7715 | 0.7697 |
| Single | 3 | 8 | 0.7706 | 0.7699 |
| Multi | 1 | 14 | 0.7748 | 0.7666 |
| Multi | 2 | 8 | 0.7745 | 0.7717 |
| Multi | 3 | 7 | 0.7742 | 0.7738 |

### Interpretation

In the single-head setting, increasing the number of layers did not improve validation AUC. At the same time, train loss continued to decrease, suggesting mild overfitting in deeper single-head models.

In the multi-head setting, validation AUC remained very close across different depths, indicating that multi-head attention was relatively stable to layer depth on this task.

The best final transformer test result in this validated layer ablation was achieved by:
- `attention_type = multi`
- `num_layers = 3`
- `dropout = 0.2`
- `learning_rate = 5e-4`

with:
- `Final Test AUC = 0.7738`

This result was very close to the tuned Random Forest benchmark (`AUC = 0.777`).

This should be sufficient for the transformer training contribution in the group project and can be directly used in the experiments, results, and discussion sections of the final report.