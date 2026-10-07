A local autonomous machine learning agent built for **CSE445 Assignment #3**.  
The project runs fully inside **Windows WSL2**, uses a local **Ollama** model for reasoning, and executes machine learning tools implemented with **scikit-learn** and **PyTorch**.

The agent follows a controlled ReAct-style workflow:

```text
User Query
    ↓
Local LLM Reasoning
    ↓
Action Selection
    ↓
ML Tool Execution
    ↓
Observation
    ↓
Controller Validation
    ↓
Retry / Self-Healing / Finalization
    ↓
Grounded Final Answer
```

The controller is responsible for enforcing required experiments, validating numerical conclusions, preventing duplicate or unsupported actions, and triggering recovery when a tool fails. The LLM is responsible for interpreting the task, selecting appropriate tools, diagnosing recoverable errors, and explaining experimental results.

---

## Features

- Fully local LLM reasoning through **Ollama**
- ReAct-style autonomous tool execution
- Dataset inspection and summary
- Classical scikit-learn model training
- Hyperparameter tuning with `GridSearchCV`
- PCA dimensionality reduction
- Sequential Feature Selection
- Deep PyTorch classification
- Batch Normalization
- Dropout regularization
- StepLR and ExponentialLR learning-rate schedulers
- Structured model-comparison reasoning
- Controller-verified numerical conclusions
- Self-healing after recoverable ML failures
- Automatic cross-validation benchmarking
- Markdown benchmark report generation
- Execution trace logging for submission and analysis

---

# Project Structure

```text
Autonomous Local LLM Machine Learning Agent
│
├── ml_tools.py
├── react_agent.py
├── benchmark_runner.py
├── requirements.txt
├── README.md
│
├── logs/
│   ├── trace_1.txt
│   ├── trace_2.txt
│   └── trace_3.txt
│
└── report/
    ├── benchmark_results.md
    └── figures/
```

### `ml_tools.py`

Contains the machine learning tools available to the agent.

Main capabilities include:

- dataset summaries
- classical model training
- PyTorch MLP training
- SVC hyperparameter tuning
- Decision Tree hyperparameter tuning
- PCA analysis
- Sequential Feature Selection
- regularized PyTorch classifiers
- learning-rate scheduler experiments

### `react_agent.py`

Implements the autonomous agent and controller.

Main responsibilities:

- communicate with the local Ollama model
- parse `Thought`, `Action`, and `Action Input`
- execute exactly one tool per action
- track completed experiments
- prevent duplicate experiments
- enforce unfinished experiment requirements
- detect tool failures
- enter self-healing mode
- validate structured recommendations
- reject unsupported numerical conclusions
- produce grounded final answers

### `benchmark_runner.py`

Runs the assignment benchmark across multiple datasets and algorithms.

It evaluates:

- Logistic Regression
- Decision Tree
- SVC

using stratified cross-validation and reports:

- fold scores
- CV mean accuracy
- CV standard deviation
- CV variance

It can also save a Markdown table to:

```text
report/benchmark_results.md
```

---

# Local LLM

The agent uses:

```text
llama3.2:3b
```

through Ollama's local REST API:

```text
http://127.0.0.1:11434/api/generate
```

No external LLM API is required.

The Python agent communicates with Ollama using the `requests` library.

---

# Environment

The project was developed in:

- Windows with WSL2
- Ubuntu under WSL
- Python virtual environment
- Ollama
- CPU-based PyTorch

Tested Python package versions:

```text
numpy==2.5.2
pandas==3.0.5
requests==2.34.2
scikit-learn==1.9.0
torch==2.13.0+cpu
```

---

# Installation

## 1. Open WSL

From Windows Terminal:

```bash
wsl
```

Move into the project directory:

```bash
cd ~/cse445_agent
```

---

## 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If the CPU-specific PyTorch build cannot be resolved automatically in a fresh environment, install the matching CPU build using the official PyTorch installation instructions and then install the remaining requirements.

---

# Ollama Setup

Install Ollama on the host system and make sure WSL can access the local Ollama service.

Check that Ollama is available:

```bash
ollama --version
```

Pull the model:

```bash
ollama pull llama3.2:3b
```

Confirm that it runs:

```bash
ollama run llama3.2:3b
```

The agent expects the Ollama generation endpoint at:

```text
http://127.0.0.1:11434/api/generate
```

---

# Machine Learning Tools

## 1. Dataset Summary

The agent can inspect supported datasets and return information such as:

- number of samples
- number of features
- number of classes
- missing-value count

Supported datasets include:

```text
iris
wine
breast_cancer
```

---

## 2. Classical Model Training

The project supports baseline scikit-learn classifiers such as:

- Logistic Regression
- Decision Tree
- Random Forest

Training results can include:

- holdout test accuracy
- cross-validation mean accuracy
- cross-validation standard deviation

---

# Hyperparameter Tuning

The function:

```python
tune_hyperparameters(...)
```

uses `GridSearchCV`.

Supported tuned models include:

- SVC
- Decision Tree

The dataset is first separated into training and holdout test sets.  
Grid search is performed only on the training portion, keeping the test set untouched for final evaluation.

Example SVC search dimensions include:

- `C`
- `kernel`
- `gamma`
- polynomial degree where applicable

Decision Tree tuning includes parameters such as:

- criterion
- maximum depth
- minimum samples split
- minimum samples leaf

The tool reports:

- best hyperparameters
- best CV mean accuracy
- best CV standard deviation
- holdout test accuracy

---

# Feature Selection and Dimensionality Reduction

The project implements two approaches.

## Principal Component Analysis

PCA reduces the feature space into a smaller set of components while retaining as much variance as possible.

The agent can compare:

- baseline accuracy
- PCA-reduced accuracy
- total explained variance

---

## Sequential Feature Selection

Sequential Feature Selection searches for a useful subset of the original features.

The implementation uses forward feature selection and reports:

- selected feature names
- baseline accuracy
- reduced-feature accuracy

Unlike PCA, SFS preserves original feature identities and can therefore be easier to interpret.

---

# Deep PyTorch Classifier

The advanced PyTorch classifier contains:

```text
Input
  ↓
Linear
  ↓
BatchNorm
  ↓
ReLU
  ↓
Dropout
  ↓
Linear
  ↓
BatchNorm
  ↓
ReLU
  ↓
Dropout
  ↓
Output Layer
```

The implementation supports configurable:

- hidden dimension
- dropout rate
- number of epochs
- learning rate
- batch size
- learning-rate scheduler

Training uses:

- `Adam`
- `CrossEntropyLoss`

The tool returns:

- final learning rate
- final training loss
- training accuracy
- test accuracy

---

# Learning-Rate Schedulers

The project supports:

## StepLR

Reduces the learning rate by a fixed factor after a specified number of epochs.

Example:

```text
initial_lr = 0.01
step_size = 25
gamma = 0.5
```

---

## ExponentialLR

Multiplies the learning rate by a constant factor after each epoch.

Example:

```text
initial_lr = 0.01
gamma = 0.95
```

The autonomous agent can run both scheduler configurations and compare their:

- final learning rates
- final losses
- training accuracies
- test accuracies

The controller validates the final numerical interpretation before accepting the recommendation.

---

# Self-Healing Agent

A major feature of the project is automatic recovery from failed machine-learning tool executions.

The controller classifies failures into categories such as:

```text
parameter_error
shape_mismatch
non_finite_loss
runtime_error
```

When a required experiment fails, the controller enters:

```text
SELF-HEALING MODE
```

The failed experiment is not marked complete.

Instead, the controller supplies the LLM with:

- failed tool name
- failed parameters
- observed error
- detected failure category
- retry count
- correction guidance

The LLM must diagnose the error and retry the same required experiment.

---

## Example: Non-Finite Loss Recovery

A controller fault-injection test deliberately replaces a valid learning rate with:

```text
1e20
```

The PyTorch tool detects:

```text
Training produced a non-finite loss.
```

The controller classifies this as:

```text
non_finite_loss
```

The agent then retries the same experiment with a safer learning rate.

Example successful retry:

```text
learning rate = 0.001
final loss    = 0.1591
test accuracy = 1.0
```

This demonstrates automatic diagnosis, correction, and recovery without manually restarting the task.

---

# Autonomous Benchmark

The benchmark evaluates three algorithms:

```text
Logistic Regression
Decision Tree
SVC
```

on two datasets:

```text
Wine
Breast Cancer
```

using:

```text
StratifiedKFold
5 folds
shuffle = True
random_state = 42
scoring = accuracy
```

For every dataset-algorithm combination, the benchmark calculates:

- fold accuracies
- CV mean accuracy
- CV standard deviation
- CV variance

This produces six experiments in total.

---

# Benchmark Results

The current benchmark produced:

| Dataset | Algorithm | CV Mean Accuracy | CV Std | CV Variance |
|---|---|---:|---:|---:|
| Wine | Logistic Regression | 0.9833 | 0.0152 | 0.000231 |
| Wine | Decision Tree | 0.8932 | 0.0425 | 0.001808 |
| Wine | SVC | 0.9833 | 0.0248 | 0.000617 |
| Breast Cancer | Logistic Regression | 0.9737 | 0.0186 | 0.000346 |
| Breast Cancer | Decision Tree | 0.9104 | 0.0312 | 0.000971 |
| Breast Cancer | SVC | 0.9772 | 0.0182 | 0.000330 |

For the **Wine** dataset, Logistic Regression and SVC tie for the highest CV mean accuracy at `0.9833`. Logistic Regression is preferred by the stability tie-break because its CV standard deviation is lower.

For the **Breast Cancer** dataset, SVC achieves the highest mean accuracy at `0.9772` and also has slightly lower variability than Logistic Regression.

Lower CV standard deviation and variance indicate more consistent performance across folds.

---

# Controller Validation

Small local language models can occasionally make arithmetic or interpretation mistakes even when the experimental observations are correct.

For that reason, the controller does not blindly accept the LLM's recommendation.

For structured comparisons it:

1. collects successful experiment outputs
2. calculates numerical relationships in Python
3. asks the LLM to select a recommendation
4. validates that recommendation against the verified evidence
5. rejects contradictory conclusions
6. requests a corrected decision when necessary
7. renders the final answer from grounded values

For the benchmark, the model-selection rule is:

```text
1. Prefer the highest CV mean accuracy.
2. If the highest mean is tied, prefer the lower CV standard deviation.
3. If both remain tied, report a tie.
```

This rule is generic and does not hardcode any particular model as the winner.

---

# Running the Agent

Activate the environment:

```bash
cd ~/cse445_agent
source venv/bin/activate
```

Then run:

```bash
python react_agent.py
```

The exact task executed depends on the prompt supplied to `run_agent_loop()`.

---

# Running the Benchmark Directly

To run the benchmark without the LLM agent:

```bash
python benchmark_runner.py
```

The Markdown output is saved to:

```text
report/benchmark_results.md
```

---

# Saving Execution Traces

The assignment requires multi-step reasoning/tool-execution traces.

A trace can be saved using `tee`:

```bash
python react_agent.py 2>&1 | tee logs/trace_3.txt
```

For custom prompts, the agent can be imported directly:

```bash
python - <<'PY' 2>&1 | tee logs/example_trace.txt
from react_agent import run_agent_loop

run_agent_loop(
    "Your machine learning task here."
)
PY
```

The current submission traces are:

```text
logs/trace_1.txt
logs/trace_2.txt
logs/trace_3.txt
```

They demonstrate:

- multi-step hyperparameter tuning
- self-healing after numerical instability
- autonomous multi-model, multi-dataset benchmarking

---

# Example ReAct Trace

A normal agent interaction follows the pattern:

```text
Thought: I need to execute the required experiment.
Action: tune_hyperparameters
Action Input: {"dataset_name": "wine", "model_type": "svc", "cv": 5}

Observation: {...}

Controller State: Required experiments still unfinished.

Thought: I will execute the remaining experiment.
Action: tune_hyperparameters
Action Input: {...}

Observation: {...}

Controller State: Entering FINALIZATION MODE.

Thought: I have gathered all necessary experimental data.
Final Answer: ...
```

The controller prevents the agent from prematurely producing a final answer before all required experiments have completed.

---

# Reliability Measures

The implementation includes several safeguards:

- exactly one tool action per LLM response
- JSON parsing and validation
- invalid-tool rejection
- duplicate experiment prevention
- required-experiment tracking
- premature-final-answer rejection
- failed experiments excluded from completion state
- bounded recovery attempts
- non-finite-loss detection
- parameter validation
- BatchNorm-related shape error handling
- numerical comparison verification
- grounded final-answer generation

These safeguards are especially important when using a relatively small local language model.

---

# Design Philosophy

The project separates responsibilities between the LLM and deterministic controller.

## LLM Responsibilities

The LLM handles:

- task interpretation
- experiment planning
- tool selection
- failure diagnosis
- trade-off interpretation
- recommendation proposals

## Controller Responsibilities

The Python controller handles:

- process enforcement
- experiment tracking
- tool execution
- error classification
- retry constraints
- arithmetic verification
- recommendation validation
- grounded final rendering

This avoids hardcoding conclusions while still protecting the system from unsupported or numerically incorrect LLM outputs.

---

# Limitations

- The local `llama3.2:3b` model can occasionally produce formatting or reasoning errors.
- The controller currently supports a defined set of ML tools rather than arbitrary Python execution.
- Experiments use relatively small built-in scikit-learn datasets.
- The benchmark currently uses accuracy as its scoring metric.
- Deep-learning experiments are designed to run locally and therefore use relatively small networks and datasets.
- CPU execution can be slower than GPU-based training for larger models or datasets.

The controller validation and self-healing mechanisms are designed to reduce the practical impact of these limitations.

---

# Requirements

```text
numpy==2.5.2
pandas==3.0.5
requests==2.34.2
scikit-learn==1.9.0
torch==2.13.0+cpu
```

---

# Summary

This project demonstrates how a small local language model can operate as an autonomous machine learning agent when combined with deterministic Python tools and controller-side safeguards.

The final system can:

- plan and execute machine learning experiments
- tune classical models
- perform feature reduction and selection
- train regularized neural networks
- compare learning-rate schedulers
- detect and recover from failed experiments
- benchmark multiple algorithms across multiple datasets
- validate LLM recommendations against numerical evidence
- generate grounded experiment summaries and Markdown reports

The result is a fully local ML-agent workflow that combines LLM reasoning with reproducible machine learning execution and deterministic validation.
