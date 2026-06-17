# Risk-Sensitive Weighted Ensemble Learning for Explainable Traffic Accident Severity Prediction under Class Imbalance

This repository contains the code and experimental outputs for a machine learning study on traffic accident severity prediction under severe class imbalance. The goal is to predict accident severity as one of three classes:

- `Slight Injury`
- `Serious Injury`
- `Fatal injury`

The project uses the public Road Traffic Accident Dataset of Addis Ababa City. The original dataset contains 12,316 accident records and 32 features. In this work, casualty-related leakage-prone columns are removed before model training, and the final experiments use 25 input predictors.

## Project Motivation

Traffic accident severity datasets are often highly imbalanced. In this dataset, most records belong to the `Slight Injury` class, while `Fatal injury` cases are very rare.

| Class | Count | Percentage |
|---|---:|---:|
| Slight Injury | 10,415 | 84.56% |
| Serious Injury | 1,743 | 14.15% |
| Fatal injury | 158 | 1.28% |

Because of this imbalance, a model can achieve high accuracy while still failing to detect serious and fatal accidents. This project therefore focuses on class-balanced and safety-sensitive metrics such as Macro-F1 and high-severity recall.

## Proposed Method

The proposed method is a risk-sensitive weighted ensemble. It combines five base classifiers:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- CatBoost

Each model receives a validation-based ensemble weight using both Macro-F1 and high-severity recall:

```text
Score_i = alpha * MacroF1_i + beta * HighSeverityRecall_i
```

where:

```text
alpha = 0.4
beta  = 0.6
```

The weighted ensemble probabilities are then adjusted using class-risk weights:

```text
Fatal injury   = 5.0
Serious Injury = 3.0
Slight Injury  = 1.0
```

This final risk adjustment gives higher decision importance to severe and fatal accident classes.

## Dataset and Preprocessing

The original dataset includes accident, driver, vehicle, road, time, and environmental variables.

The following casualty-related columns are removed to reduce target leakage risk:

```text
Casualty_class
Sex_of_casualty
Age_band_of_casualty
Casualty_severity
Work_of_casuality
Fitness_of_casuality
```

Main preprocessing steps:

1. Convert `Time` into an `Hour` feature.
2. Remove leakage-prone casualty-related columns.
3. Split data into train, validation, and test sets using a stratified 70/15/15 split.
4. Impute missing categorical values using the most frequent category.
5. Impute missing numerical values using the median.
6. One-hot encode categorical variables.
7. Standardize numerical variables.

## Experimental Results

Final test-set performance:

| Model | Accuracy | Macro-F1 | Weighted-F1 | High-Severity Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8479 | 0.3183 | 0.7813 | 0.0175 |
| Random Forest | 0.8485 | 0.3184 | 0.7816 | 0.0175 |
| Extra Trees | 0.8485 | 0.3184 | 0.7816 | 0.0175 |
| Gradient Boosting | 0.8479 | 0.3298 | 0.7862 | 0.0386 |
| CatBoost | 0.8479 | 0.3159 | 0.7803 | 0.0140 |
| **Risk-Sensitive Ensemble** | **0.8377** | **0.4387** | **0.8051** | **0.1684** |

The proposed method improves high-severity recall compared with the strongest baseline, while maintaining competitive overall accuracy.

## Figures

Place the generated figures inside an `outputs/figures/` folder, then the README will render them automatically on GitHub.

### Class Distribution

![Class Distribution](outputs/figures/class_distribution.png)

### Risk-Sensitive Ensemble Weights

![Risk-Sensitive Ensemble Weights](outputs/figures/ensemble_weights.png)

### Macro-F1 Comparison

![Macro-F1 Comparison](outputs/figures/macro_f1_comparison.png)

### High-Severity Recall Comparison

![High-Severity Recall Comparison](outputs/figures/high_severity_recall_comparison.png)

### Confusion Matrix

![Confusion Matrix](outputs/figures/confusion_matrix_risk_sensitive_ensemble.png)

### Feature Importance

![Feature Importance](outputs/figures/top_feature_importance.png)

## Suggested Repository Structure

```text
.
├── data/
│   └── RTA Dataset.csv
├── notebooks/
│   └── risk_sensitive_rta_experiment.ipynb
├── outputs/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── ensemble_weights.png
│   │   ├── macro_f1_comparison.png
│   │   ├── high_severity_recall_comparison.png
│   │   ├── confusion_matrix_risk_sensitive_ensemble.png
│   │   └── top_feature_importance.png
│   └── tables/
│       ├── validation_baseline_results.csv
│       ├── risk_tuning_results.csv
│       └── final_test_results.csv
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment and install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn catboost
```

Or, if you use a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## How to Run

1. Download the Road Traffic Accident Dataset of Addis Ababa City.
2. Place `RTA Dataset.csv` inside the `data/` folder.
3. Open the notebook in `notebooks/`.
4. Run the notebook cells in order.
5. The result tables and figures will be saved inside the `outputs/` folder.

## Main Outputs

The notebook generates:

- `validation_baseline_results.csv`
- `risk_tuning_results.csv`
- `final_test_results.csv`
- class distribution plot
- model comparison plots
- ensemble weight plot
- confusion matrix
- feature importance plot

## Notes

The method is designed as a safety-sensitive prediction framework, not as a causal explanation of accident severity. Feature importance values should be interpreted as model-level associations, not causal effects. Future work may include testing on additional datasets, adding resampling methods such as SMOTE, and evaluating calibration or SHAP-based explanations.

## Citation

If you use this repository, please cite the related paper when it becomes available.

```bibtex
@inproceedings{akbar2026risk,
  title={Risk-Sensitive Weighted Ensemble Learning for Explainable Traffic Accident Severity Prediction under Class Imbalance},
  author={Akbar, Mohammad Kaosain},
  booktitle={2026 IEEE International Conference on Future Machine Learning and Data Science},
  year={2026}
}
```

## Author

**Mohammad Kaosain Akbar**  
Department of Computer Science  
University of Calgary
