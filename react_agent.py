import json
import re

import requests

from ml_tools import AVAILABLE_TOOLS
from benchmark_runner import benchmark_models_tool

AVAILABLE_TOOLS = dict(
    AVAILABLE_TOOLS
)

AVAILABLE_TOOLS[
    "benchmark_models_tool"
] = benchmark_models_tool

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """
You are an expert Autonomous Machine Learning Assistant.

Your job is to solve machine learning tasks by selecting and
executing appropriate external tools.

You have access to these tools:

1. load_dataset_summary

   Parameters:
   {
       "dataset_name": "iris | wine | breast_cancer"
   }

   Returns:
   - number of samples
   - number of features
   - feature names
   - classes
   - missing values

2. train_sklearn_model

   Parameters:
   {
       "dataset_name": "iris | wine | breast_cancer",
       "model_type":
           "decision_tree | logistic_regression | random_forest",
       "test_size": 0.2
   }

   Returns:
   - test accuracy
   - 5-fold cross-validation mean accuracy
   - cross-validation standard deviation

3. train_pytorch_mlp

   Parameters:
   {
       "dataset_name": "iris | wine | breast_cancer",
       "hidden_dim": 32,
       "epochs": 50,
       "lr": 0.01
   }

   Returns:
   - final training loss
   - test accuracy

4. tune_hyperparameters

   Parameters:
   {
       "dataset_name": "iris | wine | breast_cancer",
       "model_type":
           "svc | decision_tree",
       "test_size": 0.2,
       "cv": 5
   }

   Returns:
   - best hyperparameters
   - best cross-validation mean accuracy
   - cross-validation standard deviation
   - test accuracy
   - number of parameter combinations tested

5. feature_selection_analysis

   Parameters:
   {
       "dataset_name": "iris | wine | breast_cancer",
       "method": "pca | sequential",
       "n_components": 2,
       "n_features_to_select": 2,
       "test_size": 0.2,
       "cv": 5
   }

   Returns for PCA:
   - original feature count
   - reduced feature count
   - explained variance ratio
   - total explained variance
   - baseline test accuracy
   - reduced test accuracy

   Returns for Sequential Feature Selection:
   - original feature count
   - selected feature count
   - selected feature names
   - baseline test accuracy
   - selected-features test accuracy

6. train_regularized_pytorch_classifier

   Parameters:
   {
       "dataset_name": "iris | wine | breast_cancer",
       "hidden_dim": 64,
       "dropout": 0.3,
       "epochs": 100,
       "lr": 0.01,
       "batch_size": 32,
       "scheduler_type":
           "none | step | exponential",
       "scheduler_step_size": 25,
       "scheduler_gamma": 0.5,
       "test_size": 0.2
   }

   Returns:
   - network architecture
   - BatchNorm configuration
   - Dropout rate
   - scheduler configuration
   - initial learning rate
   - final learning rate
   - final training loss
   - training accuracy
   - test accuracy

7. benchmark_models_tool

   Parameters:
   {
       "datasets": ["wine", "breast_cancer"],
       "algorithms": [
           "logistic_regression",
           "decision_tree",
           "svc"
       ],
       "cv": 5,
       "save_markdown": true,
       "output_path": "report/benchmark_results.md"
   }

   Runs every requested algorithm on every requested dataset
   using leakage-safe StratifiedKFold cross-validation.

   Returns:
   - fold-level accuracy scores
   - CV mean accuracy
   - CV standard deviation
   - CV variance
   - Markdown comparison table
   - Markdown output path

IMPORTANT RULES:

You operate in a strict ReAct loop.

For every response, choose EXACTLY ONE of the following modes.

===========================================================
MODE 1 - TOOL CALL
===========================================================

If you need a tool, output exactly:

Thought: Briefly explain what experiment is needed next.
Action: tool_name
Action Input: {"parameter": "value"}

Then STOP immediately.

You MUST NOT:
- output multiple Actions
- output multiple Action Inputs
- call multiple tools in one response
- write "Waiting for output"
- predict tool results
- continue reasoning after Action Input
- invent tools
- repeat successful experiments

Valid tools are ONLY:

load_dataset_summary
train_sklearn_model
train_pytorch_mlp
tune_hyperparameters
feature_selection_analysis
train_regularized_pytorch_classifier
benchmark_models_tool

TOOL SELECTION RULES:

If the user asks for:
- a model comparison benchmark
- benchmarking multiple algorithms across multiple datasets
- three algorithms across two datasets
- a cross-validation benchmark with a Markdown comparison table
- Logistic Regression, Decision Tree, and SVC evaluated together
  across Wine and Breast Cancer

use:

benchmark_models_tool

For the standard Task 3 benchmark, use:

"datasets": ["wine", "breast_cancer"]
"algorithms": [
    "logistic_regression",
    "decision_tree",
    "svc"
]
"cv": 5
"save_markdown": true
"output_path": "report/benchmark_results.md"

The benchmark tool performs all dataset/model CV combinations in
one tool execution. Do not separately call train_sklearn_model for
the models when benchmark_models_tool is the required experiment.

If the user asks to train a normal Scikit-Learn model without
hyperparameter tuning, use:

train_sklearn_model

If the user asks for:
- hyperparameter tuning
- GridSearchCV
- grid search
- tuning an SVC
- tuning an SVM
- tuning a Support Vector Machine
- tuning a Kernel SVM
- tuning a Decision Tree
- finding the best SVC parameters
- finding the best Decision Tree parameters

use:

tune_hyperparameters

For all SVC, SVM, Support Vector Machine, or Kernel SVM
hyperparameter-tuning requests, use:

"model_type": "svc"

For Decision Tree hyperparameter-tuning requests, use:

"model_type": "decision_tree"

If the user asks for:
- PCA
- Principal Component Analysis
- dimensionality reduction using PCA

use:

feature_selection_analysis

with:

"method": "pca"

If the user asks for:
- Sequential Feature Selection
- SFS
- forward sequential feature selection

use:

feature_selection_analysis

with:

"method": "sequential"

If the user explicitly gives n_components for PCA, use that value.

If the user explicitly gives n_features_to_select for Sequential
Feature Selection, use that value.

PCA and Sequential Feature Selection are separate experiments.
If both are requested, execute feature_selection_analysis twice,
ONE experiment per response.

If the user asks for:
- a deep PyTorch classifier
- a regularized PyTorch classifier
- BatchNorm
- Batch Normalization
- Dropout
- a learning-rate scheduler
- StepLR
- ExponentialLR

use:

train_regularized_pytorch_classifier

For StepLR requests, use:

"scheduler_type": "step"

For ExponentialLR requests, use:

"scheduler_type": "exponential"

If the user asks to compare StepLR and ExponentialLR,
execute train_regularized_pytorch_classifier TWICE,
ONE scheduler experiment per response.

Never execute both scheduler experiments in one response.

If the user asks to tune BOTH an SVC and a Decision Tree,
execute tune_hyperparameters TWICE, but NEVER in the same response.

You must execute required experiments ONE AT A TIME.

After each successful tool execution, the controller will tell you
which experiments remain unfinished.

When the controller gives a NEXT ACTION CONTRACT, obey it exactly.
Do not choose a different experiment. Do not repeat a completed one.

SELF-HEALING RULES:

If a required tool call fails, the controller may issue a
SELF-HEALING CONTRACT.

When that happens:
- retry the SAME required experiment
- use the SAME required tool
- preserve the required dataset and model/method/scheduler identity
- diagnose the observed error
- change only the parameter(s) needed to repair the failure
- execute exactly ONE corrected Action
- do not skip the failed experiment
- do not claim the experiment succeeded until a successful
  Observation is returned

For invalid-parameter failures, choose values permitted by the tool.

For shape/dimension failures, correct the incompatible dimension or
shape-related configuration.

For NaN/non-finite-loss failures, use a numerically safer training
configuration. Reducing an excessively large learning rate is a
reasonable first repair.

benchmark_models_tool executes the requested CV benchmark.
Final comparison and recommendation are still performed from the
successful benchmark Observation; the benchmark tool does not choose
a winner.

===========================================================
MODE 2 - FINAL ANSWER
===========================================================

Only use this mode when ALL experiments requested in the original
user query have been successfully executed.

Output:

Thought: I have gathered all necessary experimental data.
Final Answer: Provide the complete answer.

CRITICAL:

Never invent numerical results.

Every reported numerical ML result must come directly from a
successful tool Observation.

If two models have equal test accuracy, report that they are tied.

If test accuracy is tied, compare cross-validation mean accuracy
and cross-validation standard deviation.

Higher cross-validation mean accuracy indicates better average
validation performance.

Lower cross-validation standard deviation indicates more stable
performance across folds.

Do not claim that test accuracy alone proves absence of overfitting.

Begin.
"""

def query_local_llm(prompt: str) -> str:
    """
    Sends the current prompt to Ollama.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "stop": ["Observation:"],
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )

    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not connect to Ollama: {e}"
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama error: {response.text}"
        )

    return response.json().get(
        "response",
        "",
    ).strip()

def tuning_requested(query: str) -> bool:
    """
    Returns True when the query explicitly requests
    hyperparameter tuning.
    """

    tuning_terms = [
        "hyperparameter",
        "hyper-parameter",
        "gridsearch",
        "grid search",
        "gridsearchcv",
        "grid search cv",
        "randomizedsearch",
        "randomized search",
        "tune",
        "tuning",
    ]

    return any(
        term in query
        for term in tuning_terms
    )

def detect_dataset_name(user_query: str):
    """
    Returns a dataset explicitly mentioned in the user query.

    This is used only to make controller guidance more precise.
    """

    query = user_query.lower()

    if (
        "breast_cancer" in query
        or "breast cancer" in query
    ):
        return "breast_cancer"

    if "wine" in query:
        return "wine"

    if "iris" in query:
        return "iris"

    return None

def benchmark_requested(
    user_query: str,
) -> bool:
    """
    Detects comprehensive multi-model / multi-dataset benchmark
    requests so they are handled by benchmark_models_tool rather
    than as several unrelated baseline training calls.
    """

    query = user_query.lower()

    model_mentions = sum(
        [
            (
                "logistic regression" in query
                or "logistic_regression" in query
            ),
            (
                "decision tree" in query
                or "decision_tree" in query
            ),
            (
                "svc" in query
                or "svm" in query
                or "support vector" in query
            ),
        ]
    )

    dataset_mentions = sum(
        [
            "wine" in query,
            (
                "breast cancer" in query
                or "breast_cancer" in query
            ),
            "iris" in query,
        ]
    )

    explicit_benchmark_language = (
        "benchmark" in query
        or "across two datasets" in query
        or "across 2 datasets" in query
        or "model comparison" in query
        or "comparison table" in query
    )

    cv_language = (
        "cross-validation" in query
        or "cross validation" in query
        or "fold" in query
    )

    return (
        explicit_benchmark_language
        and model_mentions >= 2
        and dataset_mentions >= 2
    ) or (
        cv_language
        and model_mentions >= 3
        and dataset_mentions >= 2
    )

def detect_benchmark_datasets(
    user_query: str,
):
    """
    Extracts supported benchmark datasets in the order they
    appear in the query.
    """

    query = user_query.lower()

    candidates = []

    patterns = [
        (
            "wine",
            ["wine"],
        ),
        (
            "breast_cancer",
            [
                "breast cancer",
                "breast_cancer",
            ],
        ),
        (
            "iris",
            ["iris"],
        ),
    ]

    for canonical, aliases in patterns:

        positions = [
            query.find(alias)
            for alias in aliases
            if query.find(alias) != -1
        ]

        if positions:

            candidates.append(
                (
                    min(positions),
                    canonical,
                )
            )

    candidates.sort(
        key=lambda item: item[0]
    )

    datasets = [
        canonical
        for _, canonical in candidates
    ]

    if not datasets:

        datasets = [
            "wine",
            "breast_cancer",
        ]

    return datasets

def detect_benchmark_algorithms(
    user_query: str,
):
    """
    Extracts supported benchmark algorithms in query order.
    """

    query = user_query.lower()

    candidates = []

    patterns = [
        (
            "logistic_regression",
            [
                "logistic regression",
                "logistic_regression",
            ],
        ),
        (
            "decision_tree",
            [
                "decision tree",
                "decision_tree",
            ],
        ),
        (
            "svc",
            [
                "svc",
                "svm",
                "support vector",
            ],
        ),
    ]

    for canonical, aliases in patterns:

        positions = [
            query.find(alias)
            for alias in aliases
            if query.find(alias) != -1
        ]

        if positions:

            candidates.append(
                (
                    min(positions),
                    canonical,
                )
            )

    candidates.sort(
        key=lambda item: item[0]
    )

    algorithms = [
        canonical
        for _, canonical in candidates
    ]

    if not algorithms:

        algorithms = [
            "logistic_regression",
            "decision_tree",
            "svc",
        ]

    return algorithms

def detect_cv_folds(
    user_query: str,
    default: int = 5,
):
    """
    Detects phrases such as '5-fold' or '5 fold'.
    """

    query = user_query.lower()

    match = re.search(
        r"(\d+)\s*[- ]?\s*fold",
        query,
    )

    if match:

        return int(
            match.group(1)
        )

    return default

def validate_benchmark_parameters_against_query(
    parameters,
    user_query,
):
    """
    Ensures the benchmark call covers the datasets, algorithms,
    and fold count required by the original query.
    """

    required_datasets = (
        detect_benchmark_datasets(
            user_query
        )
    )

    required_algorithms = (
        detect_benchmark_algorithms(
            user_query
        )
    )

    required_cv = (
        detect_cv_folds(
            user_query
        )
    )

    datasets = parameters.get(
        "datasets"
    )

    algorithms = parameters.get(
        "algorithms"
    )

    cv = parameters.get(
        "cv",
        5,
    )

    if not isinstance(
        datasets,
        list,
    ):

        return False, (
            "benchmark_models_tool requires 'datasets' as a JSON list."
        )

    if not isinstance(
        algorithms,
        list,
    ):

        return False, (
            "benchmark_models_tool requires 'algorithms' as a JSON list."
        )

    normalized_datasets = [
        str(item).lower().strip()
        for item in datasets
    ]

    normalized_algorithms = [
        str(item).lower().strip()
        for item in algorithms
    ]

    if set(
        normalized_datasets
    ) != set(
        required_datasets
    ):

        return False, (
            "Benchmark datasets do not match the original query. "
            f"Required: {required_datasets}"
        )

    if set(
        normalized_algorithms
    ) != set(
        required_algorithms
    ):

        return False, (
            "Benchmark algorithms do not match the original query. "
            f"Required: {required_algorithms}"
        )

    if cv != required_cv:

        return False, (
            f"Benchmark cv must be {required_cv} for this query."
        )

    return True, ""

def detect_required_experiments(user_query: str):
    """
    Detects experiments explicitly requested by the user.

    Each requirement is stored as:
        (tool_name, model_type_or_none)
    """

    query = user_query.lower()
    requirements = []

    if benchmark_requested(
        user_query
    ):
        return [
            (
                "benchmark_models_tool",
                None,
            )
        ]

    is_tuning_request = tuning_requested(
        query
    )

    if (
        "analyze" in query
        or "analyse" in query
        or "summary" in query
        or "inspect" in query
    ):
        requirements.append(
            (
                "load_dataset_summary",
                None,
            )
        )

    if (
        "random forest" in query
        or "random_forest" in query
    ):
        requirements.append(
            (
                "train_sklearn_model",
                "random_forest",
            )
        )

    if (
        "logistic regression" in query
        or "logistic_regression" in query
    ):
        requirements.append(
            (
                "train_sklearn_model",
                "logistic_regression",
            )
        )

    decision_tree_requested = (
        "decision tree" in query
        or "decision_tree" in query
    )

    if decision_tree_requested:

        if is_tuning_request:
            requirements.append(
                (
                    "tune_hyperparameters",
                    "decision_tree",
                )
            )

        else:
            requirements.append(
                (
                    "train_sklearn_model",
                    "decision_tree",
                )
            )

    svc_requested = (
        "svc" in query
        or "svm" in query
        or "support vector" in query
        or "kernel svm" in query
        or "kernel_svm" in query
    )

    if (
        is_tuning_request
        and svc_requested
    ):
        requirements.append(
            (
                "tune_hyperparameters",
                "svc",
            )
        )

    pca_requested = (
        "pca" in query
        or "principal component" in query
    )

    if pca_requested:
        requirements.append(
            (
                "feature_selection_analysis",
                "pca",
            )
        )

    sequential_requested = (
        "sequential feature selection" in query
        or "sequential_feature_selection" in query
        or "forward sequential" in query
        or re.search(r"\\bsfs\\b", query) is not None
    )

    if sequential_requested:
        requirements.append(
            (
                "feature_selection_analysis",
                "sequential",
            )
        )

    regularized_pytorch_requested = (
        "regularized" in query
        or "batchnorm" in query
        or "batch norm" in query
        or "batch normalization" in query
        or "dropout" in query
        or "scheduler" in query
        or "steplr" in query
        or "step lr" in query
        or "exponentiallr" in query
        or "exponential lr" in query
        or "deep pytorch" in query
    )

    step_scheduler_requested = (
        "steplr" in query
        or "step lr" in query
        or "step scheduler" in query
        or "step learning" in query
    )

    exponential_scheduler_requested = (
        "exponentiallr" in query
        or "exponential lr" in query
        or "exponential scheduler" in query
        or "exponential learning" in query
    )

    if regularized_pytorch_requested:

        if step_scheduler_requested:
            requirements.append(
                (
                    "train_regularized_pytorch_classifier",
                    "step",
                )
            )

        if exponential_scheduler_requested:
            requirements.append(
                (
                    "train_regularized_pytorch_classifier",
                    "exponential",
                )
            )

        if (
            not step_scheduler_requested
            and not exponential_scheduler_requested
        ):
            requirements.append(
                (
                    "train_regularized_pytorch_classifier",
                    None,
                )
            )

    if (
        (
            "pytorch" in query
            or "mlp" in query
        )
        and not regularized_pytorch_requested
    ):
        requirements.append(
            (
                "train_pytorch_mlp",
                None,
            )
        )

    return requirements

def normalize_model_type(
    tool_name: str,
    model_type,
):
    """
    Converts equivalent model names to canonical names.
    """

    if model_type is None:
        return None

    normalized = str(
        model_type
    ).lower().strip()

    if tool_name == "tune_hyperparameters":

        if normalized in {
            "svm",
            "kernel_svm",
            "support_vector_machine",
        }:
            normalized = "svc"

    if tool_name == "feature_selection_analysis":

        if normalized in {
            "sfs",
            "sequential_feature_selection",
            "forward_sequential",
        }:
            normalized = "sequential"

        if normalized in {
            "principal_component_analysis",
            "principal_components",
        }:
            normalized = "pca"

    if tool_name == "train_regularized_pytorch_classifier":

        if normalized in {
            "steplr",
            "step_lr",
        }:
            normalized = "step"

        if normalized in {
            "exponentiallr",
            "exponential_lr",
        }:
            normalized = "exponential"

    return normalized

def experiment_matches_requirement(
    tool_name,
    parameters,
    required_tool,
    required_model,
):
    """
    Returns True when a proposed or completed tool call
    satisfies a specific required experiment.
    """

    if tool_name != required_tool:
        return False

    if required_model is None:
        return True

    parameter_name = "model_type"

    if tool_name == "feature_selection_analysis":
        parameter_name = "method"

    elif tool_name == "train_regularized_pytorch_classifier":
        parameter_name = "scheduler_type"

    executed_value = normalize_model_type(
        tool_name,
        parameters.get(parameter_name),
    )

    normalized_required = normalize_model_type(
        required_tool,
        required_model,
    )

    return executed_value == normalized_required

def find_missing_requirements(
    requirements,
    completed_experiments,
):
    """
    Returns required experiments that have not yet
    successfully completed.
    """

    missing = []

    for (
        required_tool,
        required_model,
    ) in requirements:

        requirement_found = False

        for experiment in completed_experiments:

            if experiment_matches_requirement(
                experiment["tool"],
                experiment["parameters"],
                required_tool,
                required_model,
            ):
                requirement_found = True
                break

        if not requirement_found:
            missing.append(
                (
                    required_tool,
                    required_model,
                )
            )

    return missing

def format_requirement_name(
    tool_name,
    model_name,
):
    """
    Converts one requirement into a readable name.
    """

    if tool_name == "benchmark_models_tool":
        return "multi-model cross-validation benchmark"

    if tool_name == "load_dataset_summary":
        return "dataset summary"

    if tool_name == "train_pytorch_mlp":
        return "PyTorch MLP"

    if tool_name == "train_sklearn_model":
        return str(
            model_name
        ).replace(
            "_",
            " ",
        )

    if tool_name == "tune_hyperparameters":

        if model_name == "svc":
            return "SVC hyperparameter tuning"

        if model_name == "decision_tree":
            return "Decision Tree hyperparameter tuning"

        return f"{model_name} hyperparameter tuning"

    if tool_name == "feature_selection_analysis":

        if model_name == "pca":
            return "PCA dimensionality reduction"

        if model_name == "sequential":
            return "Sequential Feature Selection"

        return f"feature-selection method: {model_name}"

    if tool_name == "train_regularized_pytorch_classifier":

        if model_name == "step":
            return "regularized PyTorch classifier with StepLR"

        if model_name == "exponential":
            return "regularized PyTorch classifier with ExponentialLR"

        return "regularized PyTorch classifier"

    return tool_name

def format_missing_requirements(
    missing,
):
    """
    Converts missing requirements into readable names.
    """

    return ", ".join(
        format_requirement_name(
            tool_name,
            model_name,
        )
        for tool_name, model_name in missing
    )

def build_next_action_contract(
    missing,
    user_query,
):
    """
    Creates precise controller guidance.

    If exactly one experiment remains, the LLM is told the exact
    tool and model type it must execute next.

    If multiple experiments remain, it must choose exactly one
    from the allowed unfinished set.
    """

    if not missing:
        return (
            "Controller State: No required experiments remain."
        )

    dataset_name = detect_dataset_name(
        user_query
    )

    benchmark_datasets = (
        detect_benchmark_datasets(
            user_query
        )
    )

    benchmark_algorithms = (
        detect_benchmark_algorithms(
            user_query
        )
    )

    benchmark_cv = (
        detect_cv_folds(
            user_query
        )
    )

    if len(missing) == 1:

        tool_name, model_name = missing[0]

        lines = [
            "NEXT ACTION CONTRACT:",
            "Exactly ONE required experiment remains.",
            f"Required tool: {tool_name}",
        ]

        if model_name is not None:

            parameter_label = "model_type"

            if tool_name == "feature_selection_analysis":
                parameter_label = "method"

            elif tool_name == "train_regularized_pytorch_classifier":
                parameter_label = "scheduler_type"

            lines.append(
                f'Required {parameter_label}: "{model_name}"'
            )

        if tool_name == "benchmark_models_tool":

            lines.append(
                "Required datasets: "
                + json.dumps(
                    benchmark_datasets
                )
            )

            lines.append(
                "Required algorithms: "
                + json.dumps(
                    benchmark_algorithms
                )
            )

            lines.append(
                f"Required cv: {benchmark_cv}"
            )

            lines.append(
                "Required save_markdown: true"
            )

            lines.append(
                'Required output_path: "report/benchmark_results.md"'
            )

        elif dataset_name is not None:

            lines.append(
                f'Required dataset_name: "{dataset_name}"'
            )

        lines.extend(
            [
                "Your next response MUST contain exactly ONE Action "
                "and ONE Action Input for this experiment.",
                "Do NOT call any other tool.",
                "Do NOT repeat any completed experiment.",
                "Do NOT output multiple Actions.",
            ]
        )

        return "\n".join(
            lines
        )

    lines = [
        "NEXT ACTION CONTRACT:",
        "Multiple required experiments remain.",
        "Choose EXACTLY ONE unfinished experiment from this list:",
    ]

    for tool_name, model_name in missing:

        description = (
            f"- tool={tool_name}"
        )

        if model_name is not None:

            parameter_label = "model_type"

            if tool_name == "feature_selection_analysis":
                parameter_label = "method"

            elif tool_name == "train_regularized_pytorch_classifier":
                parameter_label = "scheduler_type"

            description += (
                f', {parameter_label}="{model_name}"'
            )

        if tool_name == "benchmark_models_tool":

            description += (
                f", datasets={json.dumps(benchmark_datasets)}"
                f", algorithms={json.dumps(benchmark_algorithms)}"
                f", cv={benchmark_cv}"
            )

        elif dataset_name is not None:

            description += (
                f', dataset_name="{dataset_name}"'
            )

        lines.append(
            description
        )

    lines.extend(
        [
            "Execute only ONE of them in your next response.",
            "Do NOT output multiple Actions.",
            "Do NOT call a completed experiment.",
        ]
    )

    return "\n".join(
        lines
    )

def proposed_call_matches_missing(
    tool_name,
    parameters,
    missing,
):
    """
    Prevents the LLM from executing irrelevant experiments
    while required work remains.
    """

    for (
        required_tool,
        required_model,
    ) in missing:

        if experiment_matches_requirement(
            tool_name,
            parameters,
            required_tool,
            required_model,
        ):
            return True

    return False

def find_matching_requirement(
    tool_name,
    parameters,
    missing_requirements,
):
    """
    Returns the unfinished logical requirement satisfied by
    the proposed tool call.

    This lets self-healing retries stay attached to the same
    required experiment even if optional parameters change.
    """

    for (
        required_tool,
        required_model,
    ) in missing_requirements:

        if experiment_matches_requirement(
            tool_name,
            parameters,
            required_tool,
            required_model,
        ):

            return (
                required_tool,
                required_model,
            )

    return None

def classify_tool_error(
    result,
):
    """
    Classifies tool failures into categories used by the
    self-healing controller.

    Returns one of:
        parameter_error
        shape_mismatch
        non_finite_loss
        runtime_error
    """

    if isinstance(
        result,
        dict,
    ):

        error_text = str(
            result.get(
                "error",
                result,
            )
        )

    else:

        error_text = str(
            result
        )

    text = (
        error_text
        .lower()
        .strip()
    )

    non_finite_patterns = [
        "nan",
        "non-finite",
        "non finite",
        "infinite loss",
        "inf loss",
        "loss is inf",
        "loss became inf",
        "loss became nan",
        "overflow",
    ]

    if any(
        pattern in text
        for pattern in non_finite_patterns
    ):
        return "non_finite_loss"

    shape_patterns = [
        "shape mismatch",
        "shapes cannot be multiplied",
        "mat1 and mat2",
        "size mismatch",
        "dimension mismatch",
        "dimensions do not match",
        "incompatible shape",
        "incompatible shapes",
        "invalid shape",
        "expected input",
        "expected input size",
        "expected size",
        "input dimension",
        "feature dimension",
        "expected more than 1 value per channel",
        "got input size",
    ]

    if any(
        pattern in text
        for pattern in shape_patterns
    ):
        return "shape_mismatch"

    parameter_patterns = [
        "must be",
        "must contain",
        "should be",
        "unsupported",
        "invalid parameter",
        "invalid value",
        "out of range",
        "unexpected keyword",
        "unexpected argument",
        "required positional argument",
        "missing required",
        "got an unexpected keyword",
        "not found. options",
        "supported schedulers",
        "options:",
    ]

    if any(
        pattern in text
        for pattern in parameter_patterns
    ):
        return "parameter_error"

    return "runtime_error"

def build_healing_key(
    tool_name,
    requirement,
):
    """
    Identifies the logical experiment currently being repaired.

    The key deliberately ignores optional parameter values so
    changing a bad parameter does not reset the retry counter.
    """

    return (
        tool_name,
        str(
            requirement
        ),
    )

def build_self_healing_contract(
    tool_name,
    parameters,
    error_result,
    error_type,
    requirement=None,
    dataset_name=None,
    retry_number=1,
    max_retries=3,
):
    """
    Gives the LLM a constrained repair instruction.

    The controller identifies the failure class and preserves
    the logical experiment. The LLM chooses the corrected
    parameter values.
    """

    lines = [
        "SELF-HEALING CONTRACT:",
        "",
        "The previous required experiment FAILED.",
        "",
        f"Failed tool: {tool_name}",
        (
            "Failed parameters: "
            f"{json.dumps(parameters, sort_keys=True)}"
        ),
        f"Observed error: {error_result}",
        f"Detected failure category: {error_type}",
        (
            "Recovery retry: "
            f"{retry_number}/{max_retries}"
        ),
        "",
        "You MUST retry the SAME required experiment.",
        "Do not switch to another model, method, or scheduler.",
        "Do not skip the failed experiment.",
        "Do not output Final Answer.",
        "Execute exactly ONE corrected Action.",
    ]

    if requirement is not None:

        required_tool, required_model = (
            requirement
        )

        lines.extend(
            [
                "",
                f"Required tool: {required_tool}",
            ]
        )

        if required_model is not None:

            parameter_label = (
                "model_type"
            )

            if (
                required_tool
                == "feature_selection_analysis"
            ):
                parameter_label = (
                    "method"
                )

            elif (
                required_tool
                == "train_regularized_pytorch_classifier"
            ):
                parameter_label = (
                    "scheduler_type"
                )

            lines.append(
                (
                    f'Required {parameter_label}: '
                    f'"{required_model}"'
                )
            )

    if dataset_name is not None:

        lines.append(
            f'Required dataset_name: "{dataset_name}"'
        )

    lines.append(
        ""
    )

    if error_type == "parameter_error":

        lines.extend(
            [
                "Correction guidance:",
                "- Read the tool error carefully.",
                (
                    "- Correct only the parameter(s) needed "
                    "to make the call valid."
                ),
                (
                    "- Use values permitted by the tool "
                    "schema and validation rules."
                ),
            ]
        )

    elif error_type == "shape_mismatch":

        lines.extend(
            [
                "Correction guidance:",
                (
                    "- Reconsider dimensions and "
                    "shape-related parameters."
                ),
                (
                    "- Ensure model/input dimensions are "
                    "compatible with the dataset."
                ),
                (
                    "- Correct or remove the explicit "
                    "dimension value that caused the mismatch."
                ),
                (
                    "- If BatchNorm reports that it expected "
                    "more than one value per channel, then a "
                    "training batch contained only one sample."
                ),
                (
                    "- Choose a batch configuration that avoids "
                    "a singleton training batch."
                ),
                (
                    "- Do not intentionally reproduce the same "
                    "invalid one-sample batch shape."
                ),
            ]
        )

    elif error_type == "non_finite_loss":

        lines.extend(
            [
                "Correction guidance:",
                (
                    "- The training configuration became "
                    "numerically unstable."
                ),
                (
                    "- Choose a safer training "
                    "configuration."
                ),
                (
                    "- Reducing an excessively large "
                    "learning rate is normally an "
                    "appropriate first correction."
                ),
                (
                    "- Do not claim success until the retry "
                    "returns a finite result."
                ),
            ]
        )

    else:

        lines.extend(
            [
                "Correction guidance:",
                (
                    "- Read the runtime error carefully and "
                    "change only what is necessary."
                ),
                (
                    "- Preserve the same logical experiment "
                    "while repairing the failing call."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Required response format:",
            "",
            (
                "Thought: Briefly explain why the previous "
                "configuration failed and what parameter(s) "
                "you will correct."
            ),
            f"Action: {tool_name}",
            (
                "Action Input: <valid JSON for the corrected "
                "retry>"
            ),
        ]
    )

    return "\n".join(
        lines
    )

def tool_result_has_error(
    tool_result,
):
    """
    Some tools return JSON error objects rather than
    throwing Python exceptions.

    Error results must not count as completed experiments.
    """

    if not isinstance(
        tool_result,
        str,
    ):
        return False

    try:
        parsed = json.loads(
            tool_result
        )

    except json.JSONDecodeError:
        return False

    return (
        isinstance(parsed, dict)
        and "error" in parsed
    )

def extract_completed_benchmark_result(
    completed_experiments,
):
    """
    Returns the parsed successful benchmark result, if one exists.
    """

    for experiment in completed_experiments:

        if (
            experiment.get(
                "tool"
            )
            != "benchmark_models_tool"
        ):
            continue

        result = experiment.get(
            "result"
        )

        if isinstance(
            result,
            str,
        ):

            try:
                result = json.loads(
                    result
                )

            except json.JSONDecodeError:
                return None

        if (
            isinstance(
                result,
                dict,
            )
            and not result.get(
                "error"
            )
            and isinstance(
                result.get(
                    "experiments"
                ),
                list,
            )
        ):
            return result

    return None

def calculate_benchmark_recommendations(
    benchmark_result,
):
    """
    Determines the strongest algorithm per dataset using:
    1. higher CV mean accuracy;
    2. lower CV standard deviation as a tie-break;
    3. tie if both are equal.
    """

    grouped = {}

    for experiment in benchmark_result.get(
        "experiments",
        [],
    ):

        dataset = experiment.get(
            "dataset"
        )

        algorithm = experiment.get(
            "algorithm"
        )

        mean_accuracy = experiment.get(
            "cv_mean_accuracy"
        )

        std_accuracy = experiment.get(
            "cv_std_accuracy"
        )

        variance_accuracy = experiment.get(
            "cv_variance_accuracy"
        )

        if (
            dataset is None
            or algorithm is None
            or mean_accuracy is None
            or std_accuracy is None
        ):
            continue

        grouped.setdefault(
            dataset,
            [],
        ).append(
            {
                "algorithm": algorithm,
                "cv_mean_accuracy": float(
                    mean_accuracy
                ),
                "cv_std_accuracy": float(
                    std_accuracy
                ),
                "cv_variance_accuracy": (
                    float(
                        variance_accuracy
                    )
                    if variance_accuracy
                    is not None
                    else None
                ),
            }
        )

    recommendations = {}

    for dataset, rows in grouped.items():

        if not rows:
            continue

        best_mean = max(
            row[
                "cv_mean_accuracy"
            ]
            for row in rows
        )

        mean_winners = [
            row
            for row in rows
            if row[
                "cv_mean_accuracy"
            ]
            == best_mean
        ]

        if len(
            mean_winners
        ) == 1:

            winner = mean_winners[
                0
            ][
                "algorithm"
            ]

            reason = (
                "highest_cv_mean_accuracy"
            )

        else:

            best_std = min(
                row[
                    "cv_std_accuracy"
                ]
                for row in mean_winners
            )

            std_winners = [
                row
                for row in mean_winners
                if row[
                    "cv_std_accuracy"
                ]
                == best_std
            ]

            if len(
                std_winners
            ) == 1:

                winner = std_winners[
                    0
                ][
                    "algorithm"
                ]

                reason = (
                    "mean_accuracy_tie_lower_cv_std"
                )

            else:

                winner = "tie"

                reason = (
                    "mean_and_std_tie"
                )

        recommendations[
            dataset
        ] = {
            "recommendation": winner,
            "reason": reason,
            "rows": rows,
        }

    return recommendations

def build_benchmark_decision_prompt(
    user_query,
    benchmark_result,
    retry_message=None,
):
    """
    Lets the LLM identify the strongest model per dataset from
    the already-computed benchmark evidence.

    The preferred output format is intentionally strict so the
    benchmark trace remains clean and easy to validate.
    """

    table = benchmark_result.get(
        "markdown_table",
        "",
    )

    prompt = f"""
You are in BENCHMARK DECISION MODE.

The complete requested cross-validation benchmark has already run.

Original User Query:
{user_query}

Benchmark Evidence:

{table}

Metric Semantics:
- Higher CV mean accuracy means better average validation accuracy.
- Lower CV standard deviation means more stable performance across
  folds.
- Lower CV variance likewise means lower fold-to-fold variability.
- If the highest CV mean accuracy is tied on a dataset, use lower
  CV standard deviation as the stability tie-break.
- Do not infer overfitting from these CV summary statistics alone.

Your task is to identify the strongest algorithm on EACH dataset
using those rules.

OUTPUT FORMAT IS STRICT.

Your entire response MUST contain exactly TWO lines.

Line 1 MUST be exactly:

Thought: I need to compare average CV accuracy and fold stability for each dataset.

Line 2 MUST begin exactly with:

Decision:

The Decision line MUST contain exactly one valid JSON object.

The required JSON structure is:

Decision: {{"recommendations": {{"<dataset>": "<algorithm or tie>", "<dataset>": "<algorithm or tie>"}}}}

Formatting Rules:
- Do not output headings.
- Do not output Markdown tables.
- Do not output explanations.
- Do not use Markdown code fences.
- Do not output bullet points.
- Do not output Final Answer.
- Do not output Action or Action Input.
- Do not add text before the Thought line.
- Do not add text after the Decision JSON.
- Do not put the Thought inside JSON.
- Do not put the Decision inside another JSON object.
- Decision MUST be valid JSON.
- Dataset names must exactly match the benchmark.
- Recommendations must be exact algorithm names from the benchmark
  or "tie".
"""

    if retry_message:

        prompt += f"""

The previous benchmark decision was rejected.

Reason:
{retry_message}

CORRECTION REQUIRED:

Re-read the benchmark evidence and return the corrected output using
the EXACT two-line format required above.

Important correction rules:
- For each dataset, FIRST maximize CV mean accuracy.
- Only when the highest CV mean accuracy is tied, use the lower
  CV standard deviation as the stability tie-break.
- Do not choose a lower-mean algorithm merely because it has a
  lower standard deviation.
- Do not use Markdown.
- Do not use code fences.
- Do not add explanations.
- Return exactly one Thought line and one Decision line.
"""

    return prompt
def parse_benchmark_decision(
    llm_output,
):
    """
    Robustly extracts a benchmark recommendation from small local
    LLM output.

    Accepted examples include:

        Thought: ...
        Decision: {"recommendations": {...}}

    and wrapped / fenced JSON such as:

        {
          "Thought": {...},
          "Decision": {
            "recommendations": {...}
          }
        }

    Arithmetic correctness is still checked separately by
    validate_benchmark_decision().
    """

    if not isinstance(
        llm_output,
        str,
    ):

        return None, (
            "Benchmark decision output was not text."
        )

    if (
        "Action:" in llm_output
        or "Action Input:" in llm_output
        or "Final Answer:" in llm_output
    ):

        return None, (
            "Tool calls and Final Answer are forbidden in "
            "benchmark decision mode."
        )

    raw = llm_output.strip()

    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    ).strip()

    decoder = json.JSONDecoder()

    decision_label = re.search(
        r"Decision\s*:\s*",
        cleaned,
        flags=re.IGNORECASE,
    )

    if decision_label is not None:

        candidate = cleaned[
            decision_label.end():
        ].lstrip()

        json_start = candidate.find(
            "{"
        )

        if json_start != -1:

            try:

                parsed, _ = decoder.raw_decode(
                    candidate[
                        json_start:
                    ]
                )

                if isinstance(
                    parsed,
                    dict,
                ):

                    if (
                        "Decision" in parsed
                        and isinstance(
                            parsed[
                                "Decision"
                            ],
                            dict,
                        )
                    ):

                        parsed = parsed[
                            "Decision"
                        ]

                    if (
                        "decision" in parsed
                        and isinstance(
                            parsed[
                                "decision"
                            ],
                            dict,
                        )
                    ):

                        parsed = parsed[
                            "decision"
                        ]

                    if (
                        "recommendations"
                        in parsed
                    ):

                        return parsed, None

            except json.JSONDecodeError:
                pass

    for match in re.finditer(
        r"\{",
        cleaned,
    ):

        try:

            parsed, _ = decoder.raw_decode(
                cleaned[
                    match.start():
                ]
            )

        except json.JSONDecodeError:
            continue

        if not isinstance(
            parsed,
            dict,
        ):
            continue

        if (
            "Decision" in parsed
            and isinstance(
                parsed[
                    "Decision"
                ],
                dict,
            )
        ):

            decision = parsed[
                "Decision"
            ]

            if (
                "recommendations"
                in decision
            ):

                return decision, None

        if (
            "decision" in parsed
            and isinstance(
                parsed[
                    "decision"
                ],
                dict,
            )
        ):

            decision = parsed[
                "decision"
            ]

            if (
                "recommendations"
                in decision
            ):

                return decision, None

        if (
            "recommendations"
            in parsed
        ):

            return parsed, None

    recommendations_match = re.search(
        r'"recommendations"\s*:\s*'
        r'(\{[^{}]*\})',
        cleaned,
        flags=re.IGNORECASE,
    )

    if recommendations_match is not None:

        try:

            recommendations = json.loads(
                recommendations_match.group(
                    1
                )
            )

            if isinstance(
                recommendations,
                dict,
            ):

                return {
                    "recommendations": (
                        recommendations
                    )
                }, None

        except json.JSONDecodeError:
            pass

    return None, (
        "No valid benchmark Decision JSON object was found."
    )

def validate_benchmark_decision(
    decision,
    benchmark_result,
):
    """
    Validates the LLM's recommendation against the verified CV
    mean/stability comparison.
    """

    expected = (
        calculate_benchmark_recommendations(
            benchmark_result
        )
    )

    recommendations = decision.get(
        "recommendations"
    )

    if not isinstance(
        recommendations,
        dict,
    ):

        return False, (
            "Decision must contain a 'recommendations' JSON object."
        ), None

    expected_datasets = set(
        expected.keys()
    )

    supplied_datasets = set(
        recommendations.keys()
    )

    if supplied_datasets != expected_datasets:

        return False, (
            "Recommendation datasets do not exactly match the "
            f"benchmark datasets. Required: {sorted(expected_datasets)}"
        ), None

    benchmark_algorithms = set(
        benchmark_result.get(
            "algorithms",
            [],
        )
    )

    for dataset in sorted(
        expected_datasets
    ):

        supplied = str(
            recommendations[
                dataset
            ]
        ).lower().strip()

        if (
            supplied
            not in benchmark_algorithms
            and supplied != "tie"
        ):

            return False, (
                f"Invalid recommendation '{supplied}' for "
                f"dataset '{dataset}'."
            ), None

        expected_value = str(
            expected[
                dataset
            ][
                "recommendation"
            ]
        ).lower().strip()

        if supplied != expected_value:

            return False, (
                f"Recommendation for '{dataset}' contradicts "
                "the verified CV mean/stability comparison. "
                f"Verified recommendation: {expected_value}."
            ), None

    normalized = {
        "recommendations": {
            dataset: expected[
                dataset
            ][
                "recommendation"
            ]
            for dataset in expected
        },
        "details": expected,
    }

    return True, "", normalized

def render_grounded_benchmark_answer(
    benchmark_result,
    validated_decision,
    execution_history=None,
):
    """
    Renders the required Markdown table and deterministic
    evidence-based benchmark interpretation.
    """

    experiments = benchmark_result.get(
        "experiments",
        []
    )

    lines = [
        (
            "| Dataset | Algorithm | CV Mean Accuracy | "
            "CV Std | CV Variance |"
        ),
        "|---|---|---:|---:|---:|",
    ]

    for experiment in experiments:

        lines.append(
            (
                f"| {experiment['dataset']} "
                f"| {experiment['algorithm']} "
                f"| {experiment['cv_mean_accuracy']:.4f} "
                f"| {experiment['cv_std_accuracy']:.4f} "
                f"| {experiment['cv_variance_accuracy']:.6f} |"
            )
        )

    details = validated_decision[
        "details"
    ]

    interpretation_paragraphs = []

    for dataset in benchmark_result.get(
        "datasets",
        [],
    ):

        info = details[
            dataset
        ]

        rows = info[
            "rows"
        ]

        winner = info[
            "recommendation"
        ]

        reason = info[
            "reason"
        ]

        row_map = {
            row[
                "algorithm"
            ]: row
            for row in rows
        }

        if reason == "highest_cv_mean_accuracy":

            winner_row = row_map[
                winner
            ]

            most_stable_row = min(
                rows,
                key=lambda row: (
                    row[
                        "cv_std_accuracy"
                    ]
                ),
            )

            if (
                most_stable_row[
                    "algorithm"
                ]
                == winner
            ):

                stability_text = (
                    f"It also has the lowest CV standard deviation "
                    f"at {winner_row['cv_std_accuracy']:.4f} "
                    f"with variance "
                    f"{winner_row['cv_variance_accuracy']:.6f}, "
                    "so the observed accuracy and stability criteria "
                    "both favor the same algorithm."
                )

            else:

                stability_text = (
                    f"Its CV standard deviation is "
                    f"{winner_row['cv_std_accuracy']:.4f} with "
                    f"variance "
                    f"{winner_row['cv_variance_accuracy']:.6f}. "
                    f"However, {most_stable_row['algorithm']} is "
                    "more stable across folds with CV standard "
                    f"deviation "
                    f"{most_stable_row['cv_std_accuracy']:.4f}, "
                    "so there is an accuracy-versus-stability "
                    "trade-off."
                )

            interpretation_paragraphs.append(
                (
                    f"For {dataset}, {winner} is the strongest "
                    "algorithm because it has the highest CV mean "
                    f"accuracy at "
                    f"{winner_row['cv_mean_accuracy']:.4f}. "
                    + stability_text
                )
            )

        elif reason == "mean_accuracy_tie_lower_cv_std":

            winner_row = row_map[
                winner
            ]

            same_mean_rows = [
                row
                for row in rows
                if row[
                    "cv_mean_accuracy"
                ]
                == winner_row[
                    "cv_mean_accuracy"
                ]
            ]

            tied_algorithms = ", ".join(
                row[
                    "algorithm"
                ]
                for row in same_mean_rows
            )

            interpretation_paragraphs.append(
                (
                    f"For {dataset}, {tied_algorithms} tie for the "
                    "highest CV mean accuracy at "
                    f"{winner_row['cv_mean_accuracy']:.4f}. "
                    f"{winner} is preferred by the stability "
                    "tie-break because its CV standard deviation "
                    f"is lower at "
                    f"{winner_row['cv_std_accuracy']:.4f}; its CV "
                    f"variance is "
                    f"{winner_row['cv_variance_accuracy']:.6f}."
                )
            )

        else:

            interpretation_paragraphs.append(
                (
                    f"For {dataset}, the leading algorithms remain "
                    "tied on both CV mean accuracy and CV standard "
                    "deviation, so the benchmark does not establish "
                    "a single winner."
                )
            )

    output_path = benchmark_result.get(
        "markdown_output_path"
    )

    saved_text = ""

    if output_path:

        saved_text = (
            f"\n\nThe Markdown benchmark table was also saved to "
            f"`{output_path}`."
        )

    recovery_note = (
        build_recovery_note(
            execution_history
        )
    )

    recovery_text = ""

    if recovery_note:

        recovery_text = (
            "\n\n"
            + recovery_note
        )

    return (
        "Thought: I have gathered all necessary experimental data.\n\n"
        "Final Answer:\n\n"
        + "\n".join(
            lines
        )
        + "\n\n"
        + "\n\n".join(
            interpretation_paragraphs
        )
        + (
            "\n\nLower CV standard deviation and variance indicate "
            "more consistent performance across folds; they are "
            "especially useful when mean CV accuracy is tied or "
            "very close."
        )
        + saved_text
        + recovery_text
    )

def build_numeric_comparison_facts(
    completed_experiments,
):
    """
    Builds deterministic arithmetic comparisons between
    successful experiment results.

    The controller does NOT decide which experiment is better.
    It only verifies relationships such as <, >, and == so that
    the LLM cannot accidentally reverse numerical comparisons.
    """

    if len(completed_experiments) < 2:
        return ""

    parsed_experiments = []

    for experiment in completed_experiments:

        result = experiment.get(
            "result"
        )

        if isinstance(result, str):

            try:
                result = json.loads(
                    result
                )

            except json.JSONDecodeError:
                continue

        if not isinstance(
            result,
            dict,
        ):
            continue

        parsed_experiments.append(
            {
                "tool": experiment.get(
                    "tool",
                    "unknown_tool",
                ),
                "parameters": experiment.get(
                    "parameters",
                    {},
                ),
                "result": result,
            }
        )

    if len(parsed_experiments) < 2:
        return ""

    def experiment_label(
        experiment,
    ):
        """
        Creates a readable label without making any
        performance judgment.
        """

        tool_name = experiment[
            "tool"
        ]

        params = experiment[
            "parameters"
        ]

        if (
            tool_name
            == "train_regularized_pytorch_classifier"
        ):

            scheduler = str(
                params.get(
                    "scheduler_type",
                    "unknown",
                )
            ).lower()

            if scheduler == "step":
                return "StepLR"

            if scheduler == "exponential":
                return "ExponentialLR"

            return (
                f"{scheduler} scheduler"
            )

        if "model_type" in params:
            return str(
                params["model_type"]
            )

        if "method" in params:
            return str(
                params["method"]
            )

        return tool_name

    def flatten_numeric_values(
        data,
        prefix="",
    ):
        """
        Flattens nested JSON dictionaries and keeps only
        real numeric values. Booleans are deliberately ignored.
        """

        values = {}

        for key, value in data.items():

            full_key = (
                f"{prefix}.{key}"
                if prefix
                else key
            )

            if isinstance(
                value,
                dict,
            ):

                values.update(
                    flatten_numeric_values(
                        value,
                        full_key,
                    )
                )

            elif (
                isinstance(
                    value,
                    (int, float),
                )
                and not isinstance(
                    value,
                    bool,
                )
            ):

                values[
                    full_key
                ] = float(
                    value
                )

        return values

    ignored_metric_names = {
        "input_dim",
        "hidden_dim_1",
        "hidden_dim_2",
        "num_classes",
        "epochs",
        "batch_size",
        "step_size",
        "gamma",
        "test_size",
        "cv_folds",
        "n_candidates",
        "original_features",
        "reduced_features",
        "selected_feature_count",
        "selection_cv_folds",
    }

    sections = []

    for first_index in range(
        len(parsed_experiments)
    ):

        for second_index in range(
            first_index + 1,
            len(parsed_experiments),
        ):

            first = parsed_experiments[
                first_index
            ]

            second = parsed_experiments[
                second_index
            ]

            first_label = experiment_label(
                first
            )

            second_label = experiment_label(
                second
            )

            first_values = (
                flatten_numeric_values(
                    first["result"]
                )
            )

            second_values = (
                flatten_numeric_values(
                    second["result"]
                )
            )

            common_metrics = sorted(
                first_values.keys()
                & second_values.keys()
            )

            lines = []

            for metric in common_metrics:

                short_name = (
                    metric.split(".")[-1]
                )

                if (
                    short_name
                    in ignored_metric_names
                ):
                    continue

                value_a = first_values[
                    metric
                ]

                value_b = second_values[
                    metric
                ]

                if value_a < value_b:
                    relation = "<"

                elif value_a > value_b:
                    relation = ">"

                else:
                    relation = "=="

                lines.append(
                    f"- {metric}: "
                    f"{first_label} {value_a:g} "
                    f"{relation} "
                    f"{second_label} {value_b:g}"
                )

            if lines:

                sections.append(
                    (
                        f"{first_label} vs "
                        f"{second_label}:\n"
                        + "\n".join(
                            lines
                        )
                    )
                )

    if not sections:
        return ""

    return (
        "Controller-verified numerical relationships:\n\n"
        + "\n\n".join(
            sections
        )
    )

def _normalize_metric_token(
    value,
):
    """
    Normalizes an LLM-provided metric name for matching.
    """

    if value is None:
        return ""

    normalized = str(
        value
    ).lower().strip()

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalized,
    )

    return normalized.strip(
        "_"
    )

def _metric_preference(
    metric_name,
):
    """
    Returns the generally correct optimization direction for
    common ML evaluation metrics.

    This function does NOT select a model. It only states
    generic metric semantics.
    """

    short_name = (
        str(metric_name)
        .split(".")[-1]
        .lower()
    )

    if (
        "accuracy" in short_name
        or short_name in {
            "precision",
            "recall",
            "f1",
            "f1_score",
            "roc_auc",
            "auc",
            "r2",
            "r2_score",
            "explained_variance",
            "total_explained_variance",
        }
    ):
        return "higher"

    if (
        "loss" in short_name
        or "error" in short_name
        or short_name.endswith(
            "_std"
        )
        or short_name == "std"
    ):
        return "lower"

    return "neutral"

def _comparison_experiment_label(
    experiment,
):
    """
    Creates a stable readable label for one experiment.
    """

    tool_name = experiment.get(
        "tool",
        "unknown_tool",
    )

    params = experiment.get(
        "parameters",
        {},
    )

    if (
        tool_name
        == "train_regularized_pytorch_classifier"
    ):

        scheduler = str(
            params.get(
                "scheduler_type",
                "unknown",
            )
        ).lower()

        if scheduler == "step":
            return "StepLR"

        if scheduler == "exponential":
            return "ExponentialLR"

        if scheduler == "none":
            return "NoScheduler"

        return scheduler

    if "model_type" in params:
        return str(
            params["model_type"]
        )

    if "method" in params:
        method = str(
            params["method"]
        )

        if method.lower() == "pca":
            return "PCA"

        if method.lower() in {
            "sequential",
            "sfs",
            "sequential_feature_selection",
        }:
            return "Sequential Feature Selection"

        return method

    return tool_name

def _flatten_numeric_result(
    data,
    prefix="",
):
    """
    Flattens nested JSON dictionaries and keeps numeric values.
    Booleans are excluded because bool is a subclass of int.
    """

    values = {}

    if not isinstance(
        data,
        dict,
    ):
        return values

    for key, value in data.items():

        full_key = (
            f"{prefix}.{key}"
            if prefix
            else key
        )

        if isinstance(
            value,
            dict,
        ):

            values.update(
                _flatten_numeric_result(
                    value,
                    full_key,
                )
            )

        elif (
            isinstance(
                value,
                (int, float),
            )
            and not isinstance(
                value,
                bool,
            )
        ):

            values[
                full_key
            ] = float(
                value
            )

    return values

def build_comparison_records(
    completed_experiments,
):
    """
    Converts successful observations into deterministic pairwise
    numerical comparison records.

    These records contain arithmetic facts only. They do not
    contain an overall model recommendation.
    """

    parsed = []

    for experiment in completed_experiments:

        result = experiment.get(
            "result"
        )

        if isinstance(
            result,
            str,
        ):

            try:
                result = json.loads(
                    result
                )

            except json.JSONDecodeError:
                continue

        if not isinstance(
            result,
            dict,
        ):
            continue

        parsed.append(
            {
                "tool": experiment.get(
                    "tool",
                    "unknown_tool",
                ),
                "parameters": experiment.get(
                    "parameters",
                    {},
                ),
                "result": result,
                "label": (
                    _comparison_experiment_label(
                        experiment
                    )
                ),
            }
        )

    if len(parsed) < 2:
        return []

    ignored_metric_names = {
        "input_dim",
        "hidden_dim_1",
        "hidden_dim_2",
        "num_classes",
        "epochs",
        "batch_size",
        "step_size",
        "gamma",
        "test_size",
        "cv_folds",
        "n_candidates",
        "original_features",
        "reduced_features",
        "selected_feature_count",
        "selection_cv_folds",
    }

    records = []

    for first_index in range(
        len(parsed)
    ):

        for second_index in range(
            first_index + 1,
            len(parsed),
        ):

            first = parsed[
                first_index
            ]

            second = parsed[
                second_index
            ]

            first_values = (
                _flatten_numeric_result(
                    first["result"]
                )
            )

            second_values = (
                _flatten_numeric_result(
                    second["result"]
                )
            )

            common_metrics = sorted(
                first_values.keys()
                & second_values.keys()
            )

            for metric in common_metrics:

                short_name = (
                    metric.split(".")[-1]
                )

                if (
                    short_name
                    in ignored_metric_names
                ):
                    continue

                value_a = first_values[
                    metric
                ]

                value_b = second_values[
                    metric
                ]

                if value_a < value_b:
                    relation = "<"

                elif value_a > value_b:
                    relation = ">"

                else:
                    relation = "=="

                preference = (
                    _metric_preference(
                        metric
                    )
                )

                preferred_experiment = (
                    "tie"
                )

                if preference == "higher":

                    if relation == ">":
                        preferred_experiment = (
                            first["label"]
                        )

                    elif relation == "<":
                        preferred_experiment = (
                            second["label"]
                        )

                elif preference == "lower":

                    if relation == "<":
                        preferred_experiment = (
                            first["label"]
                        )

                    elif relation == ">":
                        preferred_experiment = (
                            second["label"]
                        )

                records.append(
                    {
                        "metric": metric,
                        "short_metric": short_name,
                        "experiment_a": (
                            first["label"]
                        ),
                        "value_a": value_a,
                        "relation": relation,
                        "experiment_b": (
                            second["label"]
                        ),
                        "value_b": value_b,
                        "preference": preference,
                        "preferred_experiment": (
                            preferred_experiment
                        ),
                    }
                )

    return records

def build_structured_decision_prompt(
    user_query,
    completed_experiments,
    retry_message=None,
):
    """
    Asks the LLM to make only the semantic decision:
    which metric should be primary for the user's question.

    Python already owns the arithmetic facts.
    """

    records = build_comparison_records(
        completed_experiments
    )

    results_text = []

    for index, experiment in enumerate(
        completed_experiments,
        start=1,
    ):

        results_text.append(
            (
                f"Experiment {index}\n"
                f"Label: "
                f"{_comparison_experiment_label(experiment)}\n"
                f"Tool: {experiment['tool']}\n"
                f"Parameters: "
                f"{json.dumps(experiment['parameters'], sort_keys=True)}\n"
                f"Observation: {experiment['result']}"
            )
        )

    record_lines = []

    for record in records:

        record_lines.append(
            (
                f"- metric={record['metric']}; "
                f"{record['experiment_a']}="
                f"{record['value_a']:g}; "
                f"relation={record['relation']}; "
                f"{record['experiment_b']}="
                f"{record['value_b']:g}; "
                f"preference={record['preference']}"
            )
        )

    prompt = f"""
You are in STRUCTURED COMPARISON MODE.

All requested experiments have already completed successfully.

Your job is NOT to perform arithmetic. Python has already verified
the numerical relationships.

Your job is to decide which NON-NEUTRAL evaluation metric is the
most appropriate primary metric for answering the original user's
question, and then recommend the experiment that is favored by
that metric.

Original User Query:
{user_query}

Successful Experiments:

{chr(10).join(results_text)}

Controller-Verified Metric Records:

{chr(10).join(record_lines)}

GENERAL METRIC SEMANTICS:

- Accuracy-type predictive metrics: higher is better.
- Loss/error/standard-deviation metrics: lower is better.
- Learning rate is a training state/configuration value and does
  not have a universal "better" direction.
- When the user asks which classifier or scheduler performed
  better on unseen data and held-out test accuracy is available,
  test accuracy is normally the most direct predictive metric
  unless the user explicitly prioritizes another objective.
- Do not infer overfitting or convergence from these final metrics.

Return EXACTLY:

Thought: I need to choose the primary evaluation metric from the verified evidence.
Decision: {{"primary_metric": "<exact metric name from the records>", "recommendation": "<exact experiment label or tie>", "tradeoff_metrics": ["<optional exact metric name>", "..."]}}

Rules:
- Output exactly one Thought line and one Decision line.
- Decision MUST be valid JSON.
- Do not output Final Answer.
- Do not output Action or Action Input.
- Do not invent a metric.
- Do not invent a model/scheduler label.
- Do not perform new experiments.
"""

    if retry_message:

        prompt += f"""

CORRECTION REQUIRED:

{retry_message}

Return a corrected Decision using only the verified records above.
"""

    return prompt

def parse_structured_decision(
    llm_output,
):
    """
    Extracts the JSON object following 'Decision:'.
    """

    if not isinstance(
        llm_output,
        str,
    ):
        return None, (
            "The structured decision output was not text."
        )

    if (
        "Action:" in llm_output
        or "Action Input:" in llm_output
        or "Final Answer:" in llm_output
    ):
        return None, (
            "Tool calls and Final Answer are forbidden in "
            "structured comparison mode."
        )

    match = re.search(
        r"Decision:\s*(\{[\s\S]*\})\s*$",
        llm_output.strip(),
    )

    if match is None:
        return None, (
            "No valid Decision JSON object was found."
        )

    try:
        decision = json.loads(
            match.group(1)
        )

    except json.JSONDecodeError as e:
        return None, (
            f"Decision JSON was invalid: {e}"
        )

    if not isinstance(
        decision,
        dict,
    ):
        return None, (
            "Decision must be a JSON object."
        )

    return decision, None

def validate_structured_decision(
    decision,
    completed_experiments,
):
    """
    Mathematically validates the LLM's structured comparison.

    The LLM chooses the primary metric.
    The controller verifies that the recommended experiment is
    actually favored by that metric.
    """

    records = build_comparison_records(
        completed_experiments
    )

    if not records:
        return False, (
            "No comparable numerical metrics were available."
        ), None

    primary_metric_raw = decision.get(
        "primary_metric"
    )

    recommendation_raw = decision.get(
        "recommendation"
    )

    if not primary_metric_raw:
        return False, (
            "primary_metric is required."
        ), None

    if not recommendation_raw:
        return False, (
            "recommendation is required."
        ), None

    normalized_primary = (
        _normalize_metric_token(
            primary_metric_raw
        )
    )

    matching_records = []

    for record in records:

        if normalized_primary in {
            _normalize_metric_token(
                record["metric"]
            ),
            _normalize_metric_token(
                record["short_metric"]
            ),
        }:

            matching_records.append(
                record
            )

    if not matching_records:
        available = sorted(
            {
                record["metric"]
                for record in records
            }
        )

        return False, (
            "primary_metric does not match a verified metric. "
            f"Available metrics: {available}"
        ), None

    if len(matching_records) != 1:
        return False, (
            "primary_metric is ambiguous across multiple "
            "pairwise experiment records."
        ), None

    primary_record = (
        matching_records[0]
    )

    if (
        primary_record[
            "preference"
        ]
        == "neutral"
    ):
        return False, (
            f"{primary_record['metric']} is a neutral "
            "training/configuration quantity and cannot by "
            "itself determine which experiment performed better."
        ), None

    expected_recommendation = (
        primary_record[
            "preferred_experiment"
        ]
    )

    normalized_recommendation = (
        str(
            recommendation_raw
        ).strip().lower()
    )

    if normalized_recommendation in {
        "tied",
        "equal",
        "same",
    }:
        normalized_recommendation = (
            "tie"
        )

    label_map = {
        str(
            primary_record[
                "experiment_a"
            ]
        ).lower(): (
            primary_record[
                "experiment_a"
            ]
        ),
        str(
            primary_record[
                "experiment_b"
            ]
        ).lower(): (
            primary_record[
                "experiment_b"
            ]
        ),
        "tie": "tie",
    }

    if (
        normalized_recommendation
        not in label_map
    ):
        return False, (
            "recommendation must be one of the exact experiment "
            "labels in the verified comparison, or 'tie'."
        ), None

    canonical_recommendation = (
        label_map[
            normalized_recommendation
        ]
    )

    if (
        str(
            canonical_recommendation
        ).lower()
        != str(
            expected_recommendation
        ).lower()
    ):
        return False, (
            "The recommendation contradicts the arithmetic and "
            f"metric semantics for {primary_record['metric']}. "
            f"Verified preferred result: "
            f"{expected_recommendation}."
        ), None

    tradeoff_metrics = decision.get(
        "tradeoff_metrics",
        [],
    )

    if tradeoff_metrics is None:
        tradeoff_metrics = []

    if not isinstance(
        tradeoff_metrics,
        list,
    ):
        return False, (
            "tradeoff_metrics must be a JSON list."
        ), None

    available_metric_tokens = {
        _normalize_metric_token(
            record["metric"]
        ): record["metric"]
        for record in records
    }

    available_metric_tokens.update(
        {
            _normalize_metric_token(
                record["short_metric"]
            ): record["metric"]
            for record in records
        }
    )

    canonical_tradeoffs = []

    for metric in tradeoff_metrics:

        token = _normalize_metric_token(
            metric
        )

        if token not in available_metric_tokens:
            return False, (
                f"Unknown tradeoff metric: {metric}"
            ), None

        canonical_metric = (
            available_metric_tokens[
                token
            ]
        )

        if (
            canonical_metric
            != primary_record["metric"]
            and canonical_metric
            not in canonical_tradeoffs
        ):
            canonical_tradeoffs.append(
                canonical_metric
            )

    normalized = {
        "primary_metric": (
            primary_record[
                "metric"
            ]
        ),
        "recommendation": (
            expected_recommendation
        ),
        "tradeoff_metrics": (
            canonical_tradeoffs
        ),
        "primary_record": (
            primary_record
        ),
    }

    return True, "", normalized

def _metric_display_name(
    metric,
):
    """
    Converts internal metric keys into readable labels.
    """

    short_name = (
        str(metric)
        .split(".")[-1]
    )

    names = {
        "initial_lr": "Initial learning rate",
        "final_lr": "Final learning rate",
        "final_loss": "Final loss",
        "train_accuracy": "Training accuracy",
        "test_accuracy": "Test accuracy",
        "cv_mean_accuracy": "CV mean accuracy",
        "cv_std": "CV standard deviation",
        "best_cv_mean_accuracy": (
            "Best CV mean accuracy"
        ),
        "best_cv_std": (
            "Best CV standard deviation"
        ),
        "baseline_test_accuracy": (
            "Baseline test accuracy"
        ),
        "reduced_test_accuracy": (
            "Reduced test accuracy"
        ),
        "selected_features_test_accuracy": (
            "Selected-features test accuracy"
        ),
        "total_explained_variance": (
            "Total explained variance"
        ),
    }

    return names.get(
        short_name,
        short_name.replace(
            "_",
            " ",
        ).title(),
    )

def _format_number(
    value,
):
    """
    Formats a numeric value while preserving useful precision.
    """

    value = float(
        value
    )

    if value == 0:
        return "0"

    absolute = abs(
        value
    )

    if (
        absolute < 0.0001
        or absolute >= 100000
    ):
        return f"{value:.6g}"

    if value.is_integer():
        return f"{value:.1f}"

    return f"{value:.8g}"

def render_grounded_comparison_answer(
    completed_experiments,
    validated_decision,
    execution_history=None,
):
    """
    Produces the final comparison from controller-verified facts.

    The LLM chose the primary metric and recommendation in the
    structured stage. Python renders the numbers so contradictory
    prose cannot reverse them.
    """

    records = build_comparison_records(
        completed_experiments
    )

    if not records:
        raise ValueError(
            "No verified comparison records available."
        )

    primary_record = validated_decision[
        "primary_record"
    ]

    experiment_a = (
        primary_record[
            "experiment_a"
        ]
    )

    experiment_b = (
        primary_record[
            "experiment_b"
        ]
    )

    pair_records = [
        record
        for record in records
        if (
            record[
                "experiment_a"
            ]
            == experiment_a
            and record[
                "experiment_b"
            ]
            == experiment_b
        )
    ]

    pair_records = [
        record
        for record in pair_records
        if record[
            "short_metric"
        ] != "initial_lr"
    ]

    preferred_order = {
        "final_lr": 0,
        "final_loss": 1,
        "train_accuracy": 2,
        "test_accuracy": 3,
        "best_cv_mean_accuracy": 4,
        "best_cv_std": 5,
        "cv_mean_accuracy": 6,
        "cv_std": 7,
    }

    pair_records.sort(
        key=lambda record: (
            preferred_order.get(
                record[
                    "short_metric"
                ],
                100,
            ),
            record["metric"],
        )
    )

    table_lines = [
        (
            f"| Metric | {experiment_a} | "
            f"{experiment_b} | Verified comparison |"
        ),
        "|---|---:|---:|---|",
    ]

    for record in pair_records:

        value_a = _format_number(
            record[
                "value_a"
            ]
        )

        value_b = _format_number(
            record[
                "value_b"
            ]
        )

        if record["relation"] == "==":
            comparison_text = "Equal"

        elif record["relation"] == "<":
            comparison_text = (
                f"{experiment_a} is lower"
            )

        else:
            comparison_text = (
                f"{experiment_a} is higher"
            )

        table_lines.append(
            (
                f"| {_metric_display_name(record['metric'])} "
                f"| {value_a} | {value_b} | "
                f"{comparison_text} |"
            )
        )

    recommendation = (
        validated_decision[
            "recommendation"
        ]
    )

    primary_metric_name = (
        _metric_display_name(
            primary_record[
                "metric"
            ]
        )
    )

    if recommendation == "tie":

        conclusion = (
            f"Based on the selected primary metric, "
            f"{primary_metric_name.lower()}, the two "
            "experiments are tied."
        )

    else:

        direction = (
            primary_record[
                "preference"
            ]
        )

        if direction == "higher":
            reason_phrase = (
                "the higher value"
            )

        else:
            reason_phrase = (
                "the lower value"
            )

        conclusion = (
            f"Based on the selected primary metric, "
            f"{primary_metric_name.lower()}, "
            f"{recommendation} performed better in this "
            f"experiment because it achieved {reason_phrase}."
        )

    tradeoff_sentences = []

    for record in pair_records:

        if (
            record["metric"]
            == primary_record[
                "metric"
            ]
        ):
            continue

        preference = record[
            "preference"
        ]

        if preference == "neutral":
            continue

        preferred = record[
            "preferred_experiment"
        ]

        metric_name = (
            _metric_display_name(
                record["metric"]
            ).lower()
        )

        if preferred == "tie":

            tradeoff_sentences.append(
                (
                    f"The experiments were tied on "
                    f"{metric_name}."
                )
            )

        elif (
            recommendation != "tie"
            and preferred
            != recommendation
        ):

            preference_word = (
                "higher"
                if preference
                == "higher"
                else "lower"
            )

            tradeoff_sentences.append(
                (
                    f"However, {preferred} had the "
                    f"{preference_word} {metric_name}, "
                    "so that metric favors the other "
                    "experiment."
                )
            )

    final_learning_rate_note = ""

    if any(
        record[
            "short_metric"
        ] == "final_lr"
        for record in pair_records
    ):

        final_learning_rate_note = (
            " Final learning rate is reported as a scheduler "
            "state and is not treated by itself as a performance "
            "criterion."
        )

    tradeoff_text = ""

    if tradeoff_sentences:

        tradeoff_text = (
            " "
            + " ".join(
                tradeoff_sentences
            )
        )

    recovery_note = (
        build_recovery_note(
            execution_history
        )
    )

    recovery_text = ""

    if recovery_note:
        recovery_text = (
            "\n\n"
            + recovery_note
        )

    return (
        "Thought: I have gathered all necessary experimental data.\n\n"
        "Final Answer:\n\n"
        + "\n".join(
            table_lines
        )
        + "\n\n"
        + conclusion
        + tradeoff_text
        + final_learning_rate_note
        + recovery_text
    )

def build_execution_history_summary(
    execution_history,
):
    """
    Creates a deterministic record of failed and successful tool
    attempts for finalization.

    completed_experiments remains success-only and is still used
    for requirement completion. execution_history is reporting
    evidence only.
    """

    if not execution_history:
        return "No tool execution history was recorded."

    lines = []

    for index, attempt in enumerate(
        execution_history,
        start=1,
    ):

        status = str(
            attempt.get(
                "status",
                "unknown",
            )
        ).upper()

        tool_name = attempt.get(
            "tool",
            "unknown_tool",
        )

        parameters = attempt.get(
            "parameters",
            {},
        )

        lines.append(
            f"Attempt {index}: {status}"
        )

        lines.append(
            f"Tool: {tool_name}"
        )

        lines.append(
            "Parameters: "
            + json.dumps(
                parameters,
                sort_keys=True,
            )
        )

        if status == "FAILED":

            error_type = attempt.get(
                "error_type",
                "runtime_error",
            )

            error_result = attempt.get(
                "result",
                "",
            )

            lines.append(
                f"Failure category: {error_type}"
            )

            lines.append(
                f"Observed error: {error_result}"
            )

        else:

            result = attempt.get(
                "result",
                "",
            )

            lines.append(
                f"Successful result: {result}"
            )

        lines.append("")

    return "\n".join(
        lines
    ).strip()

def build_recovery_note(
    execution_history,
):
    """
    Produces a concise controller-grounded recovery note.

    This prevents a final report from claiming that a fault
    injection or failed tool attempt never happened.
    """

    if not execution_history:
        return ""

    failures = [
        attempt
        for attempt in execution_history
        if attempt.get(
            "status"
        ) == "failed"
    ]

    successes = [
        attempt
        for attempt in execution_history
        if attempt.get(
            "status"
        ) == "success"
    ]

    if not failures:
        return ""

    first_failure = failures[0]

    failure_tool = first_failure.get(
        "tool",
        "unknown_tool",
    )

    failure_type = first_failure.get(
        "error_type",
        "runtime_error",
    )

    failure_parameters = first_failure.get(
        "parameters",
        {},
    )

    failure_result = first_failure.get(
        "result",
        "",
    )

    note = (
        "Self-healing note: "
        f"{len(failures)} failed tool attempt"
        f"{'' if len(failures) == 1 else 's'} occurred before "
        "successful completion. "
        f"The first failure used {failure_tool} with parameters "
        f"{json.dumps(failure_parameters, sort_keys=True)} and "
        f"was classified as {failure_type}. "
        f"The observed error was: {failure_result}."
    )

    if successes:

        last_success = successes[-1]

        note += (
            " The agent then completed a successful retry using "
            f"{last_success.get('tool', 'unknown_tool')} with "
            "parameters "
            f"{json.dumps(last_success.get('parameters', {}), sort_keys=True)}."
        )

    return note

def validate_final_answer_against_execution_history(
    llm_output,
    execution_history,
):
    """
    Ensures final prose does not deny a recorded failure and,
    when recovery actually occurred, acknowledges it.

    Returns:
        (True, "")
        or
        (False, reason)
    """

    if not isinstance(
        llm_output,
        str,
    ):
        return False, (
            "Final answer was not text."
        )

    failures = [
        attempt
        for attempt in (
            execution_history
            or []
        )
        if attempt.get(
            "status"
        ) == "failed"
    ]

    if not failures:
        return True, ""

    text_lower = (
        llm_output
        .lower()
    )

    denial_phrases = [
        "did not fail",
        "didn't fail",
        "no failure occurred",
        "no failures occurred",
        "there was no failure",
        "there were no failures",
        "never failed",
        "without any failure",
        "without failure",
        "fault-injection test did not fail",
        "fault injection test did not fail",
    ]

    if any(
        phrase in text_lower
        for phrase in denial_phrases
    ):
        return False, (
            "The final answer denied a tool failure that is "
            "explicitly recorded in execution_history."
        )

    recovery_terms = [
        "failed",
        "failure",
        "rejected",
        "retry",
        "retried",
        "recovery",
        "recovered",
        "self-heal",
        "self healing",
        "self-healing",
    ]

    if not any(
        term in text_lower
        for term in recovery_terms
    ):
        return False, (
            "A failed attempt occurred, but the final answer did "
            "not acknowledge the failure/recovery sequence."
        )

    return True, ""

def build_finalization_prompt(
    user_query,
    completed_experiments,
    retry_message=None,
    execution_history=None,
):
    """
    Creates a clean prompt containing only the original query
    and successful experiment results.

    This prevents the small local LLM from getting trapped in
    repeated ReAct tool-call patterns after all required tools
    have already executed.
    """

    results_text = []

    for index, experiment in enumerate(
        completed_experiments,
        start=1,
    ):

        tool_name = experiment[
            "tool"
        ]

        parameters = experiment[
            "parameters"
        ]

        result = experiment[
            "result"
        ]

        results_text.append(
            f"""
Experiment {index}

Tool:
{tool_name}

Parameters:
{json.dumps(parameters, sort_keys=True)}

Successful Observation:
{result}
""".strip()
        )

    joined_results = "\n\n".join(
        results_text
    )

    comparison_facts = (
        build_numeric_comparison_facts(
            completed_experiments
        )
    )

    execution_history_text = (
        build_execution_history_summary(
            execution_history
        )
    )

    recovery_note = (
        build_recovery_note(
            execution_history
        )
    )

    prompt = f"""
You are an expert Machine Learning Assistant.

All experiments required by the user's original request have
ALREADY been executed successfully.

You are now in FINALIZATION MODE.

YOU MUST NOT:
- call any tool
- output Action:
- output Action Input:
- request another experiment
- repeat an experiment
- output MODE 1
- invent numerical results

Use ONLY the successful experimental observations provided below.

Original User Query:
{user_query}

Successful Experimental Results:

{joined_results}

{comparison_facts}

Execution History:

{execution_history_text}

Controller-Grounded Recovery Evidence:

{recovery_note if recovery_note else "No failed tool attempt occurred before successful completion."}

Interpretation Rules:

- Answer the ORIGINAL USER QUERY directly.

- Use the successful experimental observations as the primary
  evidence for your answer.

- The Execution History is authoritative evidence about whether
  a tool attempt failed before the successful experiment.

- If Execution History contains a FAILED attempt, explicitly
  acknowledge that failure and the later successful retry.

- NEVER claim that a fault-injection test or tool execution did
  not fail when Execution History records a FAILED attempt.

- Failed attempts do not count as completed experiments, but they
  are still valid evidence when explaining self-healing behavior.

- Every numerical result, selected feature, parameter value,
  accuracy, variance value, loss value, learning-rate value,
  scheduler setting, or other experimental result that you report
  must come from the observations above.

- Carefully compare numerical values before describing one as
  higher, lower, better, worse, larger, or smaller.

- The section labeled "Controller-verified numerical relationships"
  contains deterministic arithmetic comparisons calculated from
  the successful observations. Do NOT contradict those
  relationships.

- The controller-verified relationships do NOT decide which
  experiment is better. You must still interpret their
  machine-learning significance yourself.

- If experimental results are equal, report the tie rather than
  inventing a winner.

- When interpreting classification experiments, higher test
  accuracy means higher predictive accuracy on the observed
  held-out test set, while a lower reported loss means a smaller
  value of that loss function.

- Do not treat a larger or smaller learning rate by itself as
  evidence that a model or scheduler performed better.

- When different metrics favor different experiments, explain the
  trade-off and base any recommendation on the metrics relevant to
  the original user's question.

- NEVER infer overfitting, absence of overfitting, underfitting,
  convergence, or absence of convergence unless a successful
  Observation explicitly provides evidence supporting that claim.

- Training accuracy alone cannot establish whether overfitting
  occurred, and test accuracy alone cannot establish whether
  overfitting occurred.

- Do NOT use the words "converged" or "convergence" merely because
  training accuracy reached 1.0 or because final loss is small.

- Distinguish between what the experiments directly demonstrate
  and general machine-learning interpretation.

- You may use established machine-learning concepts to explain
  the observed results, but do not invent experimental evidence.

- If the user asks for a recommendation, derive it from the
  observed results and relevant model or method trade-offs.

- If the evidence does not establish a single superior method,
  say so and explain the trade-offs instead of forcing a winner.

- Do not claim that test accuracy alone proves the presence or
  absence of overfitting.

- Do not merely repeat the observations. Interpret them and
  provide a coherent answer to the user's original question.

Before answering, internally verify that:
1. Every reported numerical value appears in an observation.
2. Numerical comparisons are mathematically correct.
3. Every required experiment is represented in the explanation.
4. The conclusion is supported by the experimental evidence.

OUTPUT FORMAT IS STRICT.

Your response MUST begin with the Thought line below.
The Thought line MUST come BEFORE Final Answer.

Output exactly:

Thought: I have gathered all necessary experimental data.
Final Answer: <write the complete evidence-based answer to the original user query>

Rules:
- Do NOT place "Final Answer:" before "Thought:".
- Do NOT add headings before the Thought line.
- Do NOT output Action: or Action Input:.
- Do NOT copy the placeholder text literally.
- After "Final Answer:", write the actual answer.
"""

    if retry_message:

        prompt += f"""

IMPORTANT CORRECTION:

Your previous finalization attempt was invalid:

{retry_message}

Do NOT call tools.

Return only the required Thought and Final Answer.
"""

    return prompt

def is_valid_final_answer_format(llm_output: str) -> bool:
    """
    Accepts a finalization response only when it follows:

    Thought: I have gathered all necessary experimental data.
    Final Answer: ...

    The Thought line must appear before Final Answer, and no
    Action / Action Input tool call may appear in finalization.
    """

    if not isinstance(llm_output, str):
        return False

    normalized = llm_output.strip()

    if (
        "Action:" in normalized
        or "Action Input:" in normalized
    ):
        return False

    pattern = re.compile(
        r"^\s*Thought:\s*"
        r"I have gathered all necessary experimental data\.\s*"
        r"Final Answer:\s*\S[\s\S]*$",
        re.IGNORECASE,
    )

    return pattern.match(normalized) is not None

def has_unsupported_ml_claims(
    llm_output: str,
) -> bool:
    """
    Rejects conclusions that the currently returned metrics
    cannot establish on their own.

    This is a generic grounding guard. It does not choose a
    winning model or scheduler.
    """

    if not isinstance(
        llm_output,
        str,
    ):
        return True

    text = llm_output.lower()

    unsupported_phrases = [
        "no overfitting",
        "not overfitting",
        "does not overfit",
        "doesn't overfit",
        "did not overfit",
        "without overfitting",
        "absence of overfitting",
        "indicating no overfitting",
        "indicates no overfitting",
        "proves no overfitting",
        "training converged",
        "model converged",
        "models converged",
        "has converged",
        "have converged",
        "indicating convergence",
        "indicates convergence",
        "proves convergence",
    ]

    return any(
        phrase in text
        for phrase in unsupported_phrases
    )

def run_agent_loop(
    user_query: str,
    max_iterations: int = 10,
    forced_first_action_overrides=None,
):
    """
    Runs the ReAct agent until all requested experiments
    complete and a valid Final Answer is produced.
    """

    print()
    print("=" * 70)
    print(f"USER QUERY: {user_query}")
    print("=" * 70)

    requirements = detect_required_experiments(
        user_query
    )

    completed_experiments = []
    executed_calls = {}

    execution_history = []

    healing_attempts = {}

    max_healing_retries = 3

    if forced_first_action_overrides is None:
        forced_first_action_overrides = {}

    forced_override_used = False

    initial_missing = find_missing_requirements(
        requirements,
        completed_experiments,
    )

    initial_contract = build_next_action_contract(
        initial_missing,
        user_query,
    )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User Query: {user_query}\n\n"
        f"{initial_contract}\n"
    )

    finalization_mode = False
    finalization_retry_message = None

    structured_decision = None
    structured_decision_retry_message = None

    benchmark_decision = None
    benchmark_decision_retry_message = None

    for step in range(
        1,
        max_iterations + 1,
    ):

        print()
        print(f"--- Step {step} ---")

        if finalization_mode:

            benchmark_result = (
                extract_completed_benchmark_result(
                    completed_experiments
                )
            )

            if benchmark_result is not None:

                if benchmark_decision is None:

                    decision_prompt = (
                        build_benchmark_decision_prompt(
                            user_query,
                            benchmark_result,
                            benchmark_decision_retry_message,
                        )
                    )

                    try:

                        llm_output = query_local_llm(
                            decision_prompt
                        )

                    except RuntimeError as e:

                        print()
                        print(
                            f"Agent error: {e}"
                        )

                        return None

                    print(
                        llm_output
                    )

                    (
                        decision,
                        parse_error,
                    ) = parse_benchmark_decision(
                        llm_output
                    )

                    if parse_error is not None:

                        benchmark_decision_retry_message = (
                            parse_error
                        )

                        observation = (
                            "Observation: Benchmark decision "
                            "rejected. "
                            f"Reason: {parse_error}"
                        )

                        print()
                        print(
                            observation
                        )

                        continue

                    (
                        valid_decision,
                        validation_error,
                        normalized_decision,
                    ) = validate_benchmark_decision(
                        decision,
                        benchmark_result,
                    )

                    if not valid_decision:

                        benchmark_decision_retry_message = (
                            validation_error
                        )

                        observation = (
                            "Observation: Benchmark decision "
                            "rejected. "
                            f"Reason: {validation_error}"
                        )

                        print()
                        print(
                            observation
                        )

                        continue

                    benchmark_decision = (
                        normalized_decision
                    )

                    print()
                    print(
                        "Controller State: Benchmark "
                        "recommendations validated against "
                        "CV mean accuracy and stability."
                    )

                    print(
                        "Controller State: Preparing "
                        "grounded Markdown benchmark answer."
                    )

                    continue

                final_output = (
                    render_grounded_benchmark_answer(
                        benchmark_result,
                        benchmark_decision,
                        execution_history=(
                            execution_history
                        ),
                    )
                )

                print(
                    final_output
                )

                print()
                print(
                    ">>> Task Completed Successfully!"
                )

                return final_output

            comparison_records = (
                build_comparison_records(
                    completed_experiments
                )
            )

            if comparison_records:

                if structured_decision is None:

                    decision_prompt = (
                        build_structured_decision_prompt(
                            user_query,
                            completed_experiments,
                            structured_decision_retry_message,
                        )
                    )

                    try:
                        llm_output = query_local_llm(
                            decision_prompt
                        )

                    except RuntimeError as e:

                        print()
                        print(
                            f"Agent error: {e}"
                        )

                        return None

                    print(
                        llm_output
                    )

                    decision, parse_error = (
                        parse_structured_decision(
                            llm_output
                        )
                    )

                    if parse_error is not None:

                        structured_decision_retry_message = (
                            parse_error
                        )

                        observation = (
                            "Observation: Structured comparison "
                            "decision rejected. "
                            f"Reason: {parse_error}"
                        )

                        print()
                        print(
                            observation
                        )

                        continue

                    (
                        is_valid_decision,
                        validation_error,
                        normalized_decision,
                    ) = validate_structured_decision(
                        decision,
                        completed_experiments,
                    )

                    if not is_valid_decision:

                        structured_decision_retry_message = (
                            validation_error
                        )

                        observation = (
                            "Observation: Structured comparison "
                            "decision rejected. "
                            f"Reason: {validation_error}"
                        )

                        print()
                        print(
                            observation
                        )

                        continue

                    structured_decision = (
                        normalized_decision
                    )

                    print()
                    print(
                        "Controller State: Structured "
                        "comparison decision validated."
                    )
                    print(
                        "Controller State: Arithmetic "
                        "relationships are grounded in the "
                        "successful tool observations."
                    )
                    print(
                        "Controller State: Preparing "
                        "grounded final answer."
                    )

                    continue

                try:
                    final_output = (
                        render_grounded_comparison_answer(
                            completed_experiments,
                            structured_decision,
                            execution_history=(
                                execution_history
                            ),
                        )
                    )

                except ValueError as e:

                    print()
                    print(
                        "Controller finalization error: "
                        f"{e}"
                    )

                    return None

                print(
                    final_output
                )

                print()
                print(
                    ">>> Task Completed Successfully!"
                )

                return final_output

            final_prompt = build_finalization_prompt(
                user_query,
                completed_experiments,
                finalization_retry_message,
                execution_history=(
                    execution_history
                ),
            )

            try:
                llm_output = query_local_llm(
                    final_prompt
                )

            except RuntimeError as e:

                print()
                print(
                    f"Agent error: {e}"
                )

                return None

            print(
                llm_output
            )

            valid_format = (
                is_valid_final_answer_format(
                    llm_output
                )
            )

            unsupported_claims = (
                has_unsupported_ml_claims(
                    llm_output
                )
            )

            (
                history_valid,
                history_error,
            ) = (
                validate_final_answer_against_execution_history(
                    llm_output,
                    execution_history,
                )
            )

            if (
                valid_format
                and not unsupported_claims
                and history_valid
            ):

                print()
                print(
                    ">>> Task Completed Successfully!"
                )

                return llm_output

            reasons = []

            if not valid_format:
                reasons.append(
                    "the required Thought/Final Answer "
                    "format was not followed"
                )

            if unsupported_claims:
                reasons.append(
                    "the answer made an unsupported claim "
                    "about overfitting or convergence"
                )

            if not history_valid:
                reasons.append(
                    history_error
                )

            reason_text = "; ".join(
                reasons
            )

            finalization_retry_message = (
                "The previous final response was rejected "
                f"because {reason_text}. "
                "Use the successful observations AND the "
                "authoritative Execution History. If a failed "
                "attempt is recorded, acknowledge the failure "
                "and successful retry rather than denying it. "
                "Do not infer overfitting or convergence from "
                "training accuracy, test accuracy, or final "
                "loss alone. Return the required Thought line "
                "followed by Final Answer. Tool calls remain "
                "forbidden during finalization."
            )

            observation = (
                "Observation: Finalization rejected. "
                f"Reason: {reason_text}. "
                "All required experiments are complete, so "
                "no additional tool execution is allowed."
            )

            print()
            print(
                observation
            )

            continue

        try:
            llm_output = query_local_llm(
                prompt
            )

        except RuntimeError as e:

            print()
            print(
                f"Agent error: {e}"
            )

            return None

        print(
            llm_output
        )

        missing_before_call = find_missing_requirements(
            requirements,
            completed_experiments,
        )

        if "Final Answer:" in llm_output:

            if missing_before_call:

                missing_text = (
                    format_missing_requirements(
                        missing_before_call
                    )
                )

                next_contract = (
                    build_next_action_contract(
                        missing_before_call,
                        user_query,
                    )
                )

                observation = (
                    "Observation: Final Answer rejected. "
                    "The original user query still has "
                    "unfinished required experiments: "
                    f"{missing_text}."
                )

                print()
                print(
                    observation
                )
                print()
                print(
                    next_contract
                )

                prompt += (
                    f"\n{llm_output}\n"
                    f"{observation}\n"
                    f"{next_contract}\n"
                )

                continue

            print()
            print(
                "Controller State: All required "
                "experiments are complete."
            )
            print(
                "Controller State: Entering "
                "FINALIZATION MODE."
            )

            finalization_mode = True
            continue

        action_matches = re.findall(
            r"Action:\s*([a-zA-Z0-9_]+)",
            llm_output,
        )

        input_matches = re.findall(
            r"Action Input:\s*(\{.*?\})",
            llm_output,
            re.DOTALL,
        )

        if (
            len(action_matches) > 1
            or len(input_matches) > 1
        ):

            next_contract = (
                build_next_action_contract(
                    missing_before_call,
                    user_query,
                )
            )

            observation = (
                "Observation: Format error. "
                "You attempted multiple tool calls "
                "in one response. "
                "Execute EXACTLY ONE tool at a time."
            )

            print()
            print(
                observation
            )
            print()
            print(
                next_contract
            )

            prompt += (
                f"\n{llm_output}\n"
                f"{observation}\n"
                f"{next_contract}\n"
            )

            continue

        if (
            len(action_matches) == 1
            and len(input_matches) == 1
        ):

            tool_name = (
                action_matches[
                    0
                ].strip()
            )

            raw_input = (
                input_matches[
                    0
                ].strip()
            )

            try:
                kwargs = json.loads(
                    raw_input
                )

            except json.JSONDecodeError as e:

                next_contract = (
                    build_next_action_contract(
                        missing_before_call,
                        user_query,
                    )
                )

                observation = (
                    "Observation: Invalid JSON in "
                    "Action Input. "
                    f"{str(e)}. "
                    "Correct the JSON and try again."
                )

                print()
                print(
                    observation
                )
                print()
                print(
                    next_contract
                )

                prompt += (
                    f"\n{llm_output}\n"
                    f"{observation}\n"
                    f"{next_contract}\n"
                )

                continue

            if (
                not forced_override_used
                and tool_name
                in forced_first_action_overrides
            ):

                overrides = (
                    forced_first_action_overrides[
                        tool_name
                    ]
                )

                if not isinstance(
                    overrides,
                    dict,
                ):

                    raise ValueError(
                        "forced_first_action_overrides "
                        "values must be dictionaries."
                    )

                original_kwargs = dict(
                    kwargs
                )

                kwargs.update(
                    overrides
                )

                forced_override_used = True

                print()
                print(
                    "Controller Test Injection: "
                    "Deliberately modifying the first "
                    "matching tool call to trigger "
                    "self-healing."
                )

                print(
                    "Controller Test Injection: "
                    f"Original parameters: "
                    f"{json.dumps(original_kwargs, sort_keys=True)}"
                )

                print(
                    "Controller Test Injection: "
                    f"Injected parameters: "
                    f"{json.dumps(kwargs, sort_keys=True)}"
                )

            call_key = (
                tool_name,
                json.dumps(
                    kwargs,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                ),
            )

            if call_key in executed_calls:

                previous_result = (
                    executed_calls[
                        call_key
                    ]
                )

                current_missing = (
                    find_missing_requirements(
                        requirements,
                        completed_experiments,
                    )
                )

                next_contract = (
                    build_next_action_contract(
                        current_missing,
                        user_query,
                    )
                )

                observation = (
                    "Observation: This exact tool "
                    "call has already been executed "
                    "successfully. It will NOT be run again.\n"
                    f"Previous result: {previous_result}"
                )

                print()
                print(
                    observation
                )
                print()
                print(
                    next_contract
                )

                prompt += (
                    f"\n{llm_output}\n"
                    f"{observation}\n"
                    f"{next_contract}\n"
                )

                continue

            if tool_name not in AVAILABLE_TOOLS:

                valid_tools = ", ".join(
                    AVAILABLE_TOOLS.keys()
                )

                next_contract = (
                    build_next_action_contract(
                        missing_before_call,
                        user_query,
                    )
                )

                observation = (
                    "Observation: "
                    f"Tool '{tool_name}' does not exist. "
                    f"Valid tools are: {valid_tools}."
                )

                print()
                print(
                    observation
                )
                print()
                print(
                    next_contract
                )

                prompt += (
                    f"\n{llm_output}\n"
                    f"{observation}\n"
                    f"{next_contract}\n"
                )

                continue

            if (
                missing_before_call
                and not proposed_call_matches_missing(
                    tool_name,
                    kwargs,
                    missing_before_call,
                )
            ):

                next_contract = (
                    build_next_action_contract(
                        missing_before_call,
                        user_query,
                    )
                )

                observation = (
                    "Observation: Tool call rejected by "
                    "the controller. This experiment does "
                    "not satisfy any currently unfinished "
                    "requirement. Do not repeat completed "
                    "or unrelated experiments."
                )

                print()
                print(
                    observation
                )
                print()
                print(
                    next_contract
                )

                prompt += (
                    f"\n{llm_output}\n"
                    f"{observation}\n"
                    f"{next_contract}\n"
                )

                continue

            if tool_name == "benchmark_models_tool":

                (
                    benchmark_params_valid,
                    benchmark_params_error,
                ) = validate_benchmark_parameters_against_query(
                    kwargs,
                    user_query,
                )

                if not benchmark_params_valid:

                    next_contract = (
                        build_next_action_contract(
                            missing_before_call,
                            user_query,
                        )
                    )

                    observation = (
                        "Observation: Benchmark tool call rejected "
                        "by the controller. "
                        f"{benchmark_params_error}"
                    )

                    print()
                    print(
                        observation
                    )
                    print()
                    print(
                        next_contract
                    )

                    prompt += (
                        f"\n{llm_output}\n"
                        f"{observation}\n"
                        f"{next_contract}\n"
                    )

                    continue

            current_requirement = (
                find_matching_requirement(
                    tool_name,
                    kwargs,
                    missing_before_call,
                )
            )

            healing_key = (
                build_healing_key(
                    tool_name,
                    current_requirement,
                )
            )

            tool_result = None
            execution_error = None

            try:

                tool_result = (
                    AVAILABLE_TOOLS[
                        tool_name
                    ](
                        **kwargs
                    )
                )

                if tool_result_has_error(
                    tool_result
                ):

                    execution_error = (
                        tool_result
                    )

            except TypeError as e:

                execution_error = (
                    "TypeError: "
                    f"{str(e)}"
                )

            except ValueError as e:

                execution_error = (
                    "ValueError: "
                    f"{str(e)}"
                )

            except Exception as e:

                execution_error = (
                    f"{type(e).__name__}: "
                    f"{str(e)}"
                )

            if execution_error is not None:

                error_type = (
                    classify_tool_error(
                        execution_error
                    )
                )

                execution_history.append(
                    {
                        "status": "failed",
                        "tool": tool_name,
                        "parameters": dict(
                            kwargs
                        ),
                        "error_type": error_type,
                        "result": execution_error,
                    }
                )

                previous_failures = (
                    healing_attempts.get(
                        healing_key,
                        0,
                    )
                )

                if (
                    previous_failures
                    >= max_healing_retries
                ):

                    observation = (
                        "Observation: Tool execution failed. "
                        f"Failure category: {error_type}. "
                        "Maximum self-healing retries were "
                        "exhausted. "
                        f"Raw error: {execution_error}"
                    )

                    print()
                    print(
                        observation
                    )

                    print()
                    print(
                        "Controller State: Maximum "
                        "self-healing retries exceeded."
                    )

                    print(
                        ">>> Task Failed After "
                        "Self-Healing Attempts"
                    )

                    return None

                retry_number = (
                    previous_failures
                    + 1
                )

                healing_attempts[
                    healing_key
                ] = retry_number

                observation = (
                    "Observation: Tool execution failed. "
                    f"Failure category: {error_type}. "
                    f"Raw error: {execution_error}"
                )

                self_healing_contract = (
                    build_self_healing_contract(
                        tool_name=tool_name,
                        parameters=kwargs,
                        error_result=execution_error,
                        error_type=error_type,
                        requirement=current_requirement,
                        dataset_name=detect_dataset_name(
                            user_query
                        ),
                        retry_number=retry_number,
                        max_retries=max_healing_retries,
                    )
                )

                print()
                print(
                    observation
                )

                print()
                print(
                    "Controller State: Entering "
                    "SELF-HEALING MODE."
                )

                print(
                    "Controller State: "
                    f"Recovery retry "
                    f"{retry_number}/"
                    f"{max_healing_retries}."
                )

                print()
                print(
                    self_healing_contract
                )

                prompt += (
                    f"\n{llm_output}\n"
                    f"{observation}\n"
                    f"{self_healing_contract}\n"
                )

                continue

            executed_calls[
                call_key
            ] = tool_result

            execution_history.append(
                {
                    "status": "success",
                    "tool": tool_name,
                    "parameters": dict(
                        kwargs
                    ),
                    "result": tool_result,
                }
            )

            completed_experiments.append(
                {
                    "tool": tool_name,
                    "parameters": kwargs,
                    "result": tool_result,
                }
            )

            healing_attempts.pop(
                healing_key,
                None,
            )

            observation = (
                "Observation: "
                f"{tool_result}"
            )

            print()
            print(
                observation
            )

            missing_after_call = (
                find_missing_requirements(
                    requirements,
                    completed_experiments,
                )
            )

            if missing_after_call:

                missing_text = (
                    format_missing_requirements(
                        missing_after_call
                    )
                )

                controller_state = (
                    "Controller State: Required "
                    "experiments still unfinished: "
                    f"{missing_text}."
                )

                next_contract = (
                    build_next_action_contract(
                        missing_after_call,
                        user_query,
                    )
                )

                print()
                print(
                    controller_state
                )
                print()
                print(
                    next_contract
                )

                prompt += (
                    f"\n{llm_output}\n"
                    f"{observation}\n"
                    f"{controller_state}\n"
                    f"{next_contract}\n"
                )

            else:

                controller_state = (
                    "Controller State: All experiments "
                    "explicitly required by the original "
                    "user query have now been completed."
                )

                print()
                print(
                    controller_state
                )

                print(
                    "Controller State: Entering "
                    "FINALIZATION MODE."
                )

                finalization_mode = True

        else:

            next_contract = (
                build_next_action_contract(
                    missing_before_call,
                    user_query,
                )
            )

            observation = (
                "Observation: Your response did not "
                "contain exactly one valid Action and "
                "Action Input. Call exactly one tool, "
                "or provide a Final Answer only when "
                "all requested experiments are complete."
            )

            print()
            print(
                observation
            )
            print()
            print(
                next_contract
            )

            prompt += (
                f"\n{llm_output}\n"
                f"{observation}\n"
                f"{next_contract}\n"
            )

    print()
    print(
        ">>> Maximum number of iterations reached."
    )

    print(
        ">>> The agent did not complete the task "
        "within the allowed number of steps."
    )

    return None

if __name__ == "__main__":

    test_task = (
        "Benchmark Logistic Regression, Decision Tree, and SVC "
        "on the Wine and Breast Cancer datasets using 5-fold "
        "cross-validation. Compare CV mean accuracy, standard "
        "deviation, and variance. Identify the strongest "
        "algorithm on each dataset, discuss the "
        "accuracy-versus-stability trade-off, and present the "
        "complete results as a Markdown table. Save the Markdown "
        "benchmark output to report/benchmark_results.md."
    )

    run_agent_loop(
        test_task
    )
