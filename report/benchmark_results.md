# Model Comparison Benchmark

Cross-validation: StratifiedKFold with 5 folds

Scoring metric: accuracy

| Dataset | Algorithm | CV Mean Accuracy | CV Std | CV Variance |
|---|---|---:|---:|---:|
| wine | logistic_regression | 0.9833 | 0.0152 | 0.000231 |
| wine | decision_tree | 0.8932 | 0.0425 | 0.001808 |
| wine | svc | 0.9833 | 0.0248 | 0.000617 |
| breast_cancer | logistic_regression | 0.9737 | 0.0186 | 0.000346 |
| breast_cancer | decision_tree | 0.9104 | 0.0312 | 0.000971 |
| breast_cancer | svc | 0.9772 | 0.0182 | 0.000330 |