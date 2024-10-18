# LFM_Entropy
An experiment applying Entropy to a Liquid Foundation Model (LFM)
The ideas for this come from https://github.com/kyegomez/LFM and https://github.com/xjdr-alt/entropix

At this point it is merely theoretical, I am trying to grok how it might work, it has not been tested.

# LFModel with Entropy Regularization

## Table of Contents

- [Introduction](#introduction)
- [Project Concept](#project-concept)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Using Poetry (Recommended)](#using-poetry-recommended)
  - [Using Setuptools and Pip](#using-setuptools-and-pip)
- [Usage](#usage)
  - [Configuration Files](#configuration-files)
  - [Running the Training Script](#running-the-training-script)
- [Project Structure](#project-structure)
- [Module and Class Summaries](#module-and-class-summaries)
  - [AdaptiveLinear](#adaptivelinear)
  - [TokenMixing](#tokenmixing)
  - [ChannelMixing](#channelmixing)
  - [MixtureOfExperts](#mixtureofexperts)
  - [LFModel](#lfmodel)
  - [Entropy Regularization Functions](#entropy-regularization-functions)
  - [Training Module](#training-module)
  - [Main Script](#main-script)
- [Conceptual Overview](#conceptual-overview)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction

This project implements the **Liquid Foundation Model (LFModel)** with integrated entropy regularization techniques. The LFModel is a custom neural network architecture designed to process sequential data, such as text or time series, by dynamically adapting its parameters based on the input. Entropy regularization is applied to encourage or discourage certain behaviors within the model, enhancing its adaptability and performance.

---

## Project Concept

The LFModel combines several innovative components:

- **AdaptiveLinear Layers**: Linear layers that adapt their weights based on an additional input, allowing dynamic adjustment during processing.
- **TokenMixing and ChannelMixing Layers**: Layers that perform adaptive mixing of tokens (sequence elements) and channels (features), capturing complex interactions.
- **Mixture of Experts (MoE) Module**: A module containing multiple experts (adaptive layers) where gating mechanisms select and combine expert outputs dynamically. Entropy regularization is applied to the gating scores to influence expert utilization.

By integrating entropy-based methods, the model can adjust its internal mechanisms to promote specialization or diversity among experts, leading to improved performance on various tasks.

---

## Installation and Setup

### Prerequisites

- **Python**: Version 3.10 or higher
- **PyTorch**: Version 2.0.0 or higher

### Clone the Repository

```bash
git clone https://github.com/davidwynter/LFModel_Entropy.git
cd LFModel_Entropy
```

### Using Poetry (Recommended)

1. **Install Poetry** (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Or refer to the [official Poetry installation guide](https://python-poetry.org/docs/#installation).

2. **Install Dependencies and the Project**:

   ```bash
   poetry install
   ```

   This will create a virtual environment, install the dependencies, and install the project in editable mode.

3. **Activate the Virtual Environment** (if not automatically activated):

   ```bash
   poetry shell
   ```

---

## Usage

### Configuration Files

Before running the training script, ensure you have the following configuration files in the project root:

- **`hyperparameters.json`**: Defines the model hyperparameters.

  ```json
  {
      "token_dim": 64,
      "channel_dim": 64,
      "expert_dim": 64,
      "adapt_dim": 32,
      "num_experts": 4,
      "lambda_entropy": 0.01
  }
  ```

- **`training.json`**: Defines the training parameters.

  ```json
  {
      "batch_size": 16,
      "sequence_length": 10,
      "num_samples": 1000,
      "num_epochs": 10,
      "learning_rate": 0.001
  }
  ```

Adjust the values as needed for your specific use case.

### Running the Training Script

**Using Poetry:**

```bash
poetry run lfmodel_train
```

Or run the main script directly:

```bash
python src/lfmodel/main.py
```

---

## Project Structure

```
LFModel_-_Entropy/
├── pyproject.toml
├── README.md
├── LICENSE
├── hyperparameters.json
├── training.json
├── src/
│   └── lfmodel/
│       ├── __init__.py
│       ├── adaptive_linear.py
│       ├── channel_mixing.py
│       ├── entropy_regularization.py
│       ├── lf_model.py
│       ├── main.py
│       ├── mixture_of_experts.py
│       ├── token_mixing.py
│       └── train.py
```

---

## Module and Class Summaries

### AdaptiveLinear

**File**: `adaptive_linear.py`

**Class**: `AdaptiveLinear`

#### Description

An extension of the standard linear (fully connected) layer that adapts its weights based on an additional input (`adapt_input`). This allows the layer to adjust its behavior dynamically in response to the input data.

#### Key Methods

- `__init__(self, in_features, out_features, adapt_dim)`: Initializes the layer with specified dimensions.
- `forward(self, x, adapt_input)`: Performs the forward pass with adaptive weight adjustment.

### TokenMixing

**File**: `token_mixing.py`

**Class**: `TokenMixing`

#### Description

Performs adaptive mixing of tokens (sequence elements) using the `AdaptiveLinear` layer. Captures relationships across the sequence dimension, allowing the model to learn complex dependencies between different positions in the sequence.

#### Key Methods

- `__init__(self, token_dim, adapt_dim)`: Initializes the token mixing layer.
- `forward(self, x, adapt_input)`: Applies adaptive token mixing.

### ChannelMixing

**File**: `channel_mixing.py`

**Class**: `ChannelMixing`

#### Description

Performs adaptive mixing of channels (features) within each token using the `AdaptiveLinear` layer. Captures inter-feature relationships, enhancing the model's representational capacity.

#### Key Methods

- `__init__(self, channel_dim, adapt_dim)`: Initializes the channel mixing layer.
- `forward(self, x, adapt_input)`: Applies adaptive channel mixing.

### MixtureOfExperts

**File**: `mixture_of_experts.py`

**Class**: `MixtureOfExperts`

#### Description

Implements a Mixture of Experts (MoE) module with entropy regularization. Contains multiple experts (adaptive layers) and uses a gating mechanism to dynamically select and combine expert outputs based on the input. Entropy regularization is applied to the gating scores to influence expert utilization.

#### Key Methods

- `__init__(self, expert_dim, num_experts, adapt_dim, lambda_entropy=0.01)`: Initializes the MoE module with specified parameters and regularization strength.
- `forward(self, x, adapt_input)`: Computes gating scores, applies entropy regularization, and combines expert outputs.

#### Entropy Regularization

- **Entropy Calculation**: Computes the entropy of the gating scores to measure uncertainty or diversity in expert selection.
- **Regularization**: The entropy term is added to the loss function, weighted by `lambda_entropy`, to encourage or discourage utilization of multiple experts.

### LFModel

**File**: `lf_model.py`

**Class**: `LFModel`

#### Description

The main model class that integrates all components: `TokenMixing`, `ChannelMixing`, and `MixtureOfExperts`. Processes sequential data by adapting its parameters based on the input, with entropy regularization applied to the MoE module.

#### Key Methods

- `__init__(self, token_dim, channel_dim, expert_dim, adapt_dim, num_experts, lambda_entropy=0.01)`: Initializes the LFModel with specified hyperparameters.
- `forward(self, x)`: Defines the forward pass through the model, including featurization, token mixing, channel mixing, MoE processing, and final output generation.

### Entropy Regularization Functions

**File**: `entropy_regularization.py`

#### Description

Contains functions for computing entropy and adjusting the loss function to include entropy regularization.

#### Key Functions

- `compute_entropy(probs)`: Computes the entropy of a probability distribution.
- `entropy_regularization_loss(model, primary_loss)`: Computes the total loss by adding the entropy regularization term to the primary loss.

### Training Module

**File**: `train.py`

#### Description

Defines the training loop, integrating entropy regularization into the optimization process.

#### Key Functions

- `train_model(model, dataloader, num_epochs, learning_rate)`: Trains the model using the specified data loader and parameters.

#### Training Workflow

1. **Data Loading**: Iterates over batches from the data loader.
2. **Forward Pass**: Computes the model's output.
3. **Loss Calculation**: Computes the primary loss and adds the entropy regularization term.
4. **Backward Pass**: Performs backpropagation.
5. **Optimization Step**: Updates the model's parameters.

### Main Script

**File**: `main.py`

#### Description

The entry point of the project. Loads configurations, initializes the model, generates dummy data (or loads real data), and starts the training process.

#### Key Components

- **Configuration Loading**: Reads hyperparameters and training parameters from `hyperparameters.json` and `training.json`.
- **Model Initialization**: Creates an instance of `LFModel` with the loaded hyperparameters.
- **Data Generation**: Generates dummy data for demonstration purposes or loads real data.
- **Training Execution**: Calls `train_model` to start the training process.

---

## Conceptual Overview

### Adaptive Mechanisms

The LFModel leverages adaptive mechanisms at multiple levels:

- **AdaptiveLinear Layers**: Adjust weights dynamically based on input, allowing the model to be more responsive and flexible.
- **Token and Channel Mixing**: Captures complex interactions across sequence positions and features by adaptively mixing tokens and channels.
- **Mixture of Experts with Entropy Regularization**: Dynamically selects experts based on input, with entropy regularization influencing the diversity or specialization of expert utilization.

### Entropy Regularization

Entropy regularization is applied to the gating scores in the MoE module:

- **Purpose**: Controls the model's behavior by encouraging or discouraging the use of multiple experts.
- **Implementation**: The entropy of the gating scores is computed and added to the loss function, weighted by a hyperparameter `lambda_entropy`.
- **Effects**:
  - **Encouraging Specialization**: Penalizing high entropy leads the model to rely on fewer experts.
  - **Encouraging Diversity**: Penalizing low entropy encourages the model to utilize multiple experts.

### Training Process

- **Data Flow**:
  1. **Input**: Receives a tensor of shape `[batch_size, sequence_length, embedding_dim]`.
  2. **Featurization**: Aggregates input to generate `adapt_input` for adaptive mechanisms.
  3. **Token Mixing**: Applies adaptive token mixing to capture sequence relationships.
  4. **Channel Mixing**: Applies adaptive channel mixing to capture feature interactions.
  5. **Mixture of Experts**: Processes data through the MoE module, with entropy regularization applied.
  6. **Output Generation**: Produces the final output through a linear layer.
- **Loss Calculation**: Combines the primary loss (e.g., MSE) with the entropy regularization term.
- **Optimization**: Updates model parameters using an optimizer (e.g., Adam).

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

