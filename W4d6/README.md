# W4D6: Responsible AI (Bias, Fairness and Transparency)

Bias audit of an income-prediction model on the UCI Adult dataset using IBM AI Fairness 360 and SHAP.

## Files
- `bias_audit.ipynb`: AIF360 disparate impact by gender, Logistic Regression, SHAP summary plot, Reweighing mitigation, before/after comparison
- `model_card.pdf` (and `model_card.md`): one-page model card
- `india_bias_case.docx`: 200-word case on Delhi Police facial recognition
- `shap_summary.png`: SHAP plot used in the model card

## Key results
| Metric | Before | After Reweighing |
|---|---|---|
| Accuracy | 0.846 | 0.840 |
| Disparate Impact | 0.316 | 0.642 |
| Stat Parity Diff | -0.183 | -0.077 |
| Avg Odds Diff | -0.093 | 0.085 |

- Original dataset disparate impact by gender is 0.363, below the 0.8 four-fifths threshold.
- `sex` ranks 4th in SHAP feature importance.
- Reweighing improved disparate impact but it is still below 0.8.

## Setup
Python 3.14 environment with `aif360`, `shap`, `scikit-learn`, `pandas<3` (AIF360 does not work with pandas 3), `matplotlib`, `tabulate`.
The UCI files `adult.data`, `adult.test` and `adult.names` must be placed in `aif360/data/raw/adult/` inside the environment.