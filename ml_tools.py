import json

import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


# ---------------------------------------------------------
# Tool 1: Dataset Summary
# ---------------------------------------------------------

def load_dataset_summary(dataset_name: str) -> str:
    """
    Loads a standard benchmark dataset and returns
    summary statistics as JSON.
    """

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": (
                f"Unknown dataset '{name}'. "
                f"Options: {list(DATASETS.keys())}"
            )
        })

    data = DATASETS[name]()

    df = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )

    df["target"] = data.target

    summary = {
        "dataset": name,
        "n_samples": df.shape[0],
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "classes": [
            str(c)
            for c in np.unique(data.target)
        ],
        "missing_values": int(
            df.isnull().sum().sum()
        ),
    }

    return json.dumps(summary)


# ---------------------------------------------------------
# Tool 2: Scikit-Learn Model Training
# ---------------------------------------------------------

def train_sklearn_model(
    dataset_name: str,
    model_type: str,
    test_size: float = 0.2,
) -> str:
    """
    Trains a Scikit-Learn model on one of the
    supported datasets.

    Supported models:
    - decision_tree
    - logistic_regression
    - random_forest
    """

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Dataset '{name}' not found."
        })

    data = DATASETS[name]()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            data.data,
            data.target,
            test_size=test_size,
            random_state=42,
            stratify=data.target,
        )
    )

    model_type = model_type.lower().strip()

    if model_type == "decision_tree":

        clf = DecisionTreeClassifier(
            max_depth=4,
            random_state=42,
        )

    elif model_type == "logistic_regression":

        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

    elif model_type == "random_forest":

        clf = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
        )

    else:
        return json.dumps({
            "error": (
                f"Unsupported model "
                f"'{model_type}'."
            )
        })

    # Train model
    clf.fit(X_train, y_train)

    # Test-set predictions
    preds = clf.predict(X_test)

    # Test accuracy
    acc = accuracy_score(
        y_test,
        preds,
    )

    # 5-fold cross-validation
    cv_scores = cross_val_score(
        clf,
        data.data,
        data.target,
        cv=5,
    )

    result = {
        "model": model_type,
        "dataset": name,
        "test_accuracy": round(
            float(acc),
            4,
        ),
        "cv_mean_accuracy": round(
            float(cv_scores.mean()),
            4,
        ),
        "cv_std": round(
            float(cv_scores.std()),
            4,
        ),
    }

    return json.dumps(result)


# ---------------------------------------------------------
# Tool 3: PyTorch MLP
# ---------------------------------------------------------

def train_pytorch_mlp(
    dataset_name: str,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 0.01,
) -> str:
    """
    Trains a simple PyTorch Multilayer Perceptron
    on the selected classification dataset.
    """

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Dataset '{name}' not found."
        })

    data = DATASETS[name]()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42,
            stratify=data.target,
        )
    )

    # -----------------------------------------------------
    # Feature Standardization
    # -----------------------------------------------------

    mean = X_train.mean(axis=0)

    std = (
        X_train.std(axis=0)
        + 1e-7
    )

    X_train = (
        X_train - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std

    # -----------------------------------------------------
    # Dataset dimensions
    # -----------------------------------------------------

    num_features = X_train.shape[1]

    num_classes = len(
        np.unique(data.target)
    )

    # -----------------------------------------------------
    # Convert NumPy arrays to PyTorch tensors
    # -----------------------------------------------------

    X_t = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_t = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    X_val_t = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    y_val_t = torch.tensor(
        y_test,
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # Neural Network
    # -----------------------------------------------------

    model = nn.Sequential(
        nn.Linear(
            num_features,
            hidden_dim,
        ),
        nn.ReLU(),
        nn.Linear(
            hidden_dim,
            num_classes,
        ),
    )

    # -----------------------------------------------------
    # Loss and optimizer
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(epochs):

        optimizer.zero_grad()

        output = model(X_t)

        loss = criterion(
            output,
            y_t,
        )

        loss.backward()

        optimizer.step()

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    with torch.no_grad():

        test_output = model(
            X_val_t
        )

        test_predictions = torch.argmax(
            test_output,
            dim=1,
        )

        accuracy = (
            (
                test_predictions
                == y_val_t
            )
            .float()
            .mean()
            .item()
        )

    result = {
        "framework": "PyTorch",
        "dataset": name,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "final_loss": round(
            float(loss.item()),
            4,
        ),
        "test_accuracy": round(
            float(accuracy),
            4,
        ),
    }

    return json.dumps(result)


def tune_hyperparameters(
    dataset_name: str,
    model_type: str,
    test_size: float = 0.2,
    cv: int = 5,
) -> str:
    """
    Performs hyperparameter tuning using GridSearchCV.

    Supported models:
        - svc
        - decision_tree

    The dataset is first divided into training and test sets.
    GridSearchCV is performed only on the training set so that
    the test set remains unseen until final evaluation.
    """

    # -------------------------------------------------------
    # Validate dataset
    # -------------------------------------------------------
    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": (
                f"Dataset '{name}' not found. "
                f"Options: {list(DATASETS.keys())}"
            )
        })

    # -------------------------------------------------------
    # Validate general parameters
    # -------------------------------------------------------
    if not 0 < test_size < 1:
        return json.dumps({
            "error": "test_size must be between 0 and 1."
        })

    if cv < 2:
        return json.dumps({
            "error": "cv must be at least 2."
        })

    # -------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------
    data = DATASETS[name]()

    X = data.data
    y = data.target

    # -------------------------------------------------------
    # Hold-out test set
    #
    # GridSearchCV will only see X_train/y_train.
    # X_test/y_test stay completely unseen until the end.
    # -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    model_type = model_type.lower().strip()

    # -------------------------------------------------------
    # SVC / Kernel SVM
    # -------------------------------------------------------
    if model_type in {"svc", "svm", "kernel_svm"}:

        # Scaling is important for SVMs.
        #
        # StandardScaler is placed inside a Pipeline so that
        # scaling occurs independently inside each CV fold.
        # This avoids data leakage.
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC()),
        ])

        # Separate dictionaries make the grid logically cleaner:
        # linear kernel does not need gamma/degree,
        # RBF uses gamma,
        # polynomial uses degree.
        param_grid = [
            {
                "svc__kernel": ["linear"],
                "svc__C": [0.1, 1.0, 10.0],
            },
            {
                "svc__kernel": ["rbf"],
                "svc__C": [0.1, 1.0, 10.0],
                "svc__gamma": ["scale", 0.01, 0.1],
            },
            {
                "svc__kernel": ["poly"],
                "svc__C": [0.1, 1.0, 10.0],
                "svc__degree": [2, 3],
                "svc__gamma": ["scale"],
            },
        ]

        canonical_model_name = "svc"

    # -------------------------------------------------------
    # Decision Tree
    # -------------------------------------------------------
    elif model_type == "decision_tree":

        estimator = DecisionTreeClassifier(
            random_state=42
        )

        param_grid = {
            "criterion": [
                "gini",
                "entropy",
            ],
            "max_depth": [
                None,
                3,
                5,
                8,
            ],
            "min_samples_split": [
                2,
                5,
                10,
            ],
            "min_samples_leaf": [
                1,
                2,
            ],
        }

        canonical_model_name = "decision_tree"

    else:
        return json.dumps({
            "error": (
                f"Unsupported model '{model_type}'. "
                "Supported models: svc, decision_tree."
            )
        })

    # -------------------------------------------------------
    # Grid Search
    # -------------------------------------------------------
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    # -------------------------------------------------------
    # Final evaluation on untouched test set
    # -------------------------------------------------------
    best_model = grid_search.best_estimator_

    test_predictions = best_model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    # -------------------------------------------------------
    # Extract CV statistics of best configuration
    # -------------------------------------------------------
    best_index = grid_search.best_index_

    best_cv_std = grid_search.cv_results_[
        "std_test_score"
    ][best_index]

    best_params = grid_search.best_params_.copy()

    # Pipeline parameters appear as:
    #
    # svc__C
    # svc__kernel
    # svc__gamma
    #
    # Remove "svc__" so the LLM gets cleaner JSON.
    if canonical_model_name == "svc":
        best_params = {
            key.replace("svc__", ""): value
            for key, value in best_params.items()
        }

    number_of_candidates = len(
        grid_search.cv_results_["params"]
    )

    # -------------------------------------------------------
    # Return JSON observation for ReAct agent
    # -------------------------------------------------------
    result = {
        "search_method": "GridSearchCV",
        "dataset": name,
        "model": canonical_model_name,
        "cv_folds": cv,
        "test_size": test_size,
        "n_candidates": number_of_candidates,
        "best_params": best_params,
        "best_cv_mean_accuracy": round(
            float(grid_search.best_score_),
            4,
        ),
        "best_cv_std": round(
            float(best_cv_std),
            4,
        ),
        "test_accuracy": round(
            float(test_accuracy),
            4,
        ),
    }

    return json.dumps(result)

def feature_selection_analysis(
    dataset_name: str,
    method: str,
    n_components: int = 2,
    n_features_to_select: int = 2,
    test_size: float = 0.2,
    cv: int = 5,
) -> str:
    """
    Performs feature selection or dimensionality reduction.

    Supported methods:
        - pca
        - sequential

    PCA:
        StandardScaler -> PCA -> LogisticRegression

    Sequential Feature Selection:
        StandardScaler -> SequentialFeatureSelector
        -> LogisticRegression

    A baseline Logistic Regression model using all original
    features is also evaluated for comparison.
    """

    # -----------------------------------------------------
    # Validate dataset
    # -----------------------------------------------------

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": (
                f"Dataset '{name}' not found. "
                f"Options: {list(DATASETS.keys())}"
            )
        })

    # -----------------------------------------------------
    # Validate common parameters
    # -----------------------------------------------------

    if not 0 < test_size < 1:
        return json.dumps({
            "error": "test_size must be between 0 and 1."
        })

    if cv < 2:
        return json.dumps({
            "error": "cv must be at least 2."
        })

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    data = DATASETS[name]()

    X = data.data
    y = data.target

    feature_names = [
        str(feature)
        for feature in data.feature_names
    ]

    original_feature_count = X.shape[1]

    # -----------------------------------------------------
    # Train/Test Split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    # -----------------------------------------------------
    # Baseline model using all original features
    # -----------------------------------------------------

    baseline_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    baseline_pipeline.fit(
        X_train,
        y_train,
    )

    baseline_predictions = baseline_pipeline.predict(
        X_test
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    method = method.lower().strip()

    # =====================================================
    # PCA
    # =====================================================

    if method == "pca":

        if not isinstance(
            n_components,
            int,
        ):
            return json.dumps({
                "error": (
                    "n_components must be an integer."
                )
            })

        if (
            n_components < 1
            or n_components > original_feature_count
        ):
            return json.dumps({
                "error": (
                    "n_components must be between "
                    f"1 and {original_feature_count}."
                )
            })

        pca_pipeline = Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=n_components,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ])

        pca_pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pca_pipeline.predict(
            X_test
        )

        reduced_accuracy = accuracy_score(
            y_test,
            predictions,
        )

        fitted_pca = pca_pipeline.named_steps[
            "pca"
        ]

        explained_variance_ratio = [
            round(
                float(value),
                4,
            )
            for value in (
                fitted_pca.explained_variance_ratio_
            )
        ]

        total_explained_variance = float(
            fitted_pca
            .explained_variance_ratio_
            .sum()
        )

        return json.dumps({
            "method": "pca",
            "dataset": name,
            "original_features": (
                original_feature_count
            ),
            "reduced_features": n_components,
            "explained_variance_ratio": (
                explained_variance_ratio
            ),
            "total_explained_variance": round(
                total_explained_variance,
                4,
            ),
            "baseline_test_accuracy": round(
                float(baseline_accuracy),
                4,
            ),
            "reduced_test_accuracy": round(
                float(reduced_accuracy),
                4,
            ),
        })

    # =====================================================
    # Sequential Feature Selection
    # =====================================================

    elif method in {
        "sequential",
        "sfs",
        "sequential_feature_selection",
    }:

        if not isinstance(
            n_features_to_select,
            int,
        ):
            return json.dumps({
                "error": (
                    "n_features_to_select must "
                    "be an integer."
                )
            })

        if (
            n_features_to_select < 1
            or n_features_to_select
            >= original_feature_count
        ):
            return json.dumps({
                "error": (
                    "n_features_to_select must be "
                    "at least 1 and smaller than the "
                    "number of original features "
                    f"({original_feature_count})."
                )
            })

        selector_estimator = (
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            )
        )

        sfs_pipeline = Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "selector",
                SequentialFeatureSelector(
                    estimator=selector_estimator,
                    n_features_to_select=(
                        n_features_to_select
                    ),
                    direction="forward",
                    scoring="accuracy",
                    cv=cv,
                    n_jobs=-1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ])

        sfs_pipeline.fit(
            X_train,
            y_train,
        )

        predictions = sfs_pipeline.predict(
            X_test
        )

        reduced_accuracy = accuracy_score(
            y_test,
            predictions,
        )

        fitted_selector = (
            sfs_pipeline.named_steps[
                "selector"
            ]
        )

        support_mask = (
            fitted_selector.get_support()
        )

        selected_features = [
            feature_names[index]
            for index, selected
            in enumerate(support_mask)
            if selected
        ]

        return json.dumps({
            "method": (
                "sequential_feature_selection"
            ),
            "dataset": name,
            "direction": "forward",
            "selection_cv_folds": cv,
            "original_features": (
                original_feature_count
            ),
            "selected_feature_count": (
                n_features_to_select
            ),
            "selected_features": (
                selected_features
            ),
            "baseline_test_accuracy": round(
                float(baseline_accuracy),
                4,
            ),
            "selected_features_test_accuracy": round(
                float(reduced_accuracy),
                4,
            ),
        })

    # =====================================================
    # Unsupported Method
    # =====================================================

    else:

        return json.dumps({
            "error": (
                f"Unsupported method '{method}'. "
                "Supported methods: "
                "pca, sequential."
            )
        })

class RegularizedMLP(nn.Module):
    """
    Deeper PyTorch MLP with Batch Normalization and Dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()

        second_hidden_dim = max(
            hidden_dim // 2,
            2,
        )

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.BatchNorm1d(
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                second_hidden_dim,
            ),
            nn.BatchNorm1d(
                second_hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                second_hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.network(
            x
        )

def train_regularized_pytorch_classifier(
    dataset_name: str,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 32,
    scheduler_type: str = "step",
    scheduler_step_size: int = 25,
    scheduler_gamma: float = 0.5,
    test_size: float = 0.2,
) -> str:
    """
    Trains a deeper PyTorch classifier with:

    - Batch Normalization
    - Dropout
    - Mini-batch training
    - Configurable learning-rate scheduler

    Supported scheduler types:
        - none
        - step
        - exponential
    """

    # -----------------------------------------------------
    # Validate dataset
    # -----------------------------------------------------

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": (
                f"Dataset '{name}' not found. "
                f"Options: {list(DATASETS.keys())}"
            )
        })

    # -----------------------------------------------------
    # Validate parameters
    # -----------------------------------------------------

    if hidden_dim < 4:
        return json.dumps({
            "error": (
                "hidden_dim must be at least 4."
            )
        })

    if not 0 <= dropout < 1:
        return json.dumps({
            "error": (
                "dropout must be between "
                "0.0 and 1.0."
            )
        })

    if epochs < 1:
        return json.dumps({
            "error": (
                "epochs must be at least 1."
            )
        })

    if lr <= 0:
        return json.dumps({
            "error": (
                "lr must be greater than 0."
            )
        })

    if batch_size < 2:
        return json.dumps({
            "error": (
                "batch_size must be at least 2."
            )
        })

    if not 0 < test_size < 1:
        return json.dumps({
            "error": (
                "test_size must be between 0 and 1."
            )
        })

    if scheduler_step_size < 1:
        return json.dumps({
            "error": (
                "scheduler_step_size must "
                "be at least 1."
            )
        })

    if not 0 < scheduler_gamma <= 1:
        return json.dumps({
            "error": (
                "scheduler_gamma must be "
                "greater than 0 and at most 1."
            )
        })

    scheduler_type = (
        scheduler_type
        .lower()
        .strip()
    )

    if scheduler_type not in {
        "none",
        "step",
        "exponential",
    }:
        return json.dumps({
            "error": (
                f"Unsupported scheduler_type "
                f"'{scheduler_type}'. "
                "Supported schedulers: "
                "none, step, exponential."
            )
        })

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    np.random.seed(
        42
    )

    torch.manual_seed(
        42
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            42
        )

    # -----------------------------------------------------
    # Select device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    data = DATASETS[name]()

    X = data.data.astype(
        np.float32
    )

    y = data.target.astype(
        np.int64
    )

    # -----------------------------------------------------
    # Train/Test split
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    # -----------------------------------------------------
    # Standardization
    #
    # Statistics come ONLY from training data.
    # -----------------------------------------------------

    mean = X_train.mean(
        axis=0
    )

    std = (
        X_train.std(
            axis=0
        )
        + 1e-7
    )

    X_train = (
        X_train - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std

    # -----------------------------------------------------
    # Convert to tensors
    # -----------------------------------------------------

    X_train_t = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_t = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    X_test_t = torch.tensor(
        X_test,
        dtype=torch.float32,
    ).to(
        device
    )

    y_test_t = torch.tensor(
        y_test,
        dtype=torch.long,
    ).to(
        device
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    train_dataset = TensorDataset(
        X_train_t,
        y_train_t,
    )

    generator = torch.Generator()

    generator.manual_seed(
        42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    # -----------------------------------------------------
    # Model dimensions
    # -----------------------------------------------------

    input_dim = X_train.shape[1]

    num_classes = len(
        np.unique(y)
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    model = RegularizedMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(
        device
    )

    # -----------------------------------------------------
    # Loss function
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    initial_lr = float(
        optimizer.param_groups[0][
            "lr"
        ]
    )

    # -----------------------------------------------------
    # Learning-rate scheduler
    # -----------------------------------------------------

    scheduler = None

    if scheduler_type == "step":

        scheduler = (
            optim.lr_scheduler.StepLR(
                optimizer,
                step_size=(
                    scheduler_step_size
                ),
                gamma=(
                    scheduler_gamma
                ),
            )
        )

    elif (
        scheduler_type
        == "exponential"
    ):

        scheduler = (
            optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=(
                    scheduler_gamma
                ),
            )
        )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    final_loss = None

    for epoch in range(
        epochs
    ):

        model.train()

        epoch_loss = 0.0

        total_samples = 0

        for (
            batch_X,
            batch_y,
        ) in train_loader:

            batch_X = batch_X.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                batch_X
            )

            loss = criterion(
                outputs,
                batch_y,
            )

            # ---------------------------------------------
            # Detect unstable / NaN training
            # ---------------------------------------------

            if not torch.isfinite(
                loss
            ):
                return json.dumps({
                    "error": (
                        "Training produced a "
                        "non-finite loss."
                    )
                })

            loss.backward()

            optimizer.step()

            batch_count = (
                batch_X.size(0)
            )

            epoch_loss += (
                loss.item()
                * batch_count
            )

            total_samples += (
                batch_count
            )

        final_loss = (
            epoch_loss
            / total_samples
        )

        # ---------------------------------------------
        # Scheduler updates once per epoch
        # ---------------------------------------------

        if scheduler is not None:
            scheduler.step()

    # -----------------------------------------------------
    # Final learning rate
    # -----------------------------------------------------

    final_lr = float(
        optimizer.param_groups[0][
            "lr"
        ]
    )

    # -----------------------------------------------------
    # Training-set evaluation
    # -----------------------------------------------------

    model.eval()

    X_train_eval = X_train_t.to(
        device
    )

    y_train_eval = y_train_t.to(
        device
    )

    with torch.no_grad():

        train_outputs = model(
            X_train_eval
        )

        train_predictions = (
            torch.argmax(
                train_outputs,
                dim=1,
            )
        )

        train_accuracy = (
            (
                train_predictions
                == y_train_eval
            )
            .float()
            .mean()
            .item()
        )

        # ---------------------------------------------
        # Test-set evaluation
        # ---------------------------------------------

        test_outputs = model(
            X_test_t
        )

        test_predictions = (
            torch.argmax(
                test_outputs,
                dim=1,
            )
        )

        test_accuracy = (
            (
                test_predictions
                == y_test_t
            )
            .float()
            .mean()
            .item()
        )

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    result = {
        "framework": "PyTorch",
        "model": "regularized_mlp",
        "dataset": name,
        "device": str(device),

        "architecture": {
            "input_dim": (
                input_dim
            ),
            "hidden_dim_1": (
                hidden_dim
            ),
            "hidden_dim_2": (
                max(
                    hidden_dim // 2,
                    2,
                )
            ),
            "num_classes": (
                num_classes
            ),
            "batch_norm": True,
            "dropout": (
                dropout
            ),
        },

        "training": {
            "epochs": (
                epochs
            ),
            "batch_size": (
                batch_size
            ),
            "optimizer": "Adam",
        },

        "scheduler": {
            "type": (
                scheduler_type
            ),
            "step_size": (
                scheduler_step_size
                if scheduler_type
                == "step"
                else None
            ),
            "gamma": (
                scheduler_gamma
                if scheduler_type
                != "none"
                else None
            ),
            "initial_lr": round(
                initial_lr,
                8,
            ),
            "final_lr": round(
                final_lr,
                8,
            ),
        },

        "final_loss": round(
            float(final_loss),
            4,
        ),

        "train_accuracy": round(
            float(train_accuracy),
            4,
        ),

        "test_accuracy": round(
            float(test_accuracy),
            4,
        ),
    }

    return json.dumps(
        result
    )

# ---------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------

AVAILABLE_TOOLS = {
    "load_dataset_summary":
        load_dataset_summary,

    "train_sklearn_model":
        train_sklearn_model,

    "train_pytorch_mlp":
        train_pytorch_mlp,

    "tune_hyperparameters":
        tune_hyperparameters,
    
    "feature_selection_analysis": 
        feature_selection_analysis,

    "train_regularized_pytorch_classifier":
        train_regularized_pytorch_classifier,
}