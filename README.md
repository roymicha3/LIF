# LIF (Leaky Integrate-and-Fire) Neural Network Framework

A comprehensive framework for building, training, and analyzing Leaky Integrate-and-Fire neural networks. This framework provides a flexible architecture for spiking neural network experiments with configurable components and extensive monitoring capabilities.

## Project Structure

```
LIF/
├── analysis/          # Analysis and visualization tools
├── data/             # Dataset storage and management
├── encoders/         # Neural spike encoding implementations
├── experiment/       # Experiment framework
│   ├── db/          # Experiment database management
│   └── trial.py     # Trial implementation
├── network/          # Neural network components
│   ├── activation/   # Activation functions and spike generators
│   ├── kernel/       # Network kernels (DEN, etc.)
│   ├── learning/     # Learning rules and algorithms
│   ├── loss/        # Loss functions
│   ├── optimizer/   # Optimization algorithms
│   └── topology/    # Network architecture definitions
├── pipeline/         # Training and evaluation pipelines
│   ├── callback/    # Training callbacks and monitoring
│   └── training_pipeline.py  # Main training implementation
├── settings/        # Configuration and environment settings
└── tools/           # Utility functions and helpers
```

## Core Components

### 1. Neural Network Architecture

- **Kernel Types**:
  - DENKernel (Dynamic Evolving Neural Kernel)
  - Configurable parameters: tau_s, tau_m, learning rates

- **Learning Rules**:
  - SingleSpikeLR (Single Spike Learning Rule)
  - Customizable thresholds and learning parameters

- **Optimizers**:
  - MomentumOptimizer
  - Configurable learning rates and momentum

### 2. Experiment Framework

The experiment system is designed for systematic neural network research:

```yaml
# experiment.yaml
name: "experiment_name"
desc: "Experiment description"
base_dir: "./outputs"
db_path: "./outputs/db/experiment.db"

trials:
  - name: "trial_1"
    repeat: 1
    settings:
      dataset:
        len: 1000
  # Additional trials...
```

- **Database Integration**: Automatic logging of experiments and trials
- **Trial Management**: Support for multiple trials with different parameters
- **Parameter Sweeping**: Systematic exploration of hyperparameters

### 3. Training Pipeline

The training pipeline (`TrainingPipeline`) provides a comprehensive training framework:

- **Data Management**:
  - Automatic train/validation/test splitting
  - Configurable batch sizes
  - Dataset normalization and preprocessing

- **Training Features**:
  - Epoch-based training with progress monitoring
  - Automatic device management (CPU/GPU)
  - Learning rate scheduling
  - Early stopping capabilities

- **Evaluation**:
  - Per-epoch validation
  - Label-wise accuracy tracking
  - Loss monitoring and optimization

### 4. Callback System

Extensive callback system for monitoring and controlling the training process:

- **MetricsTracker**: Tracks and logs training metrics
- **EarlyStopping**: Prevents overfitting with configurable patience
- **MLflowCallback**: Integration with MLflow for experiment tracking
- **ModelCheckpoint**: Saves model states during training

```yaml
# Pipeline configuration with callbacks
pipeline:
  type: "TrainingPipeline"
  epochs: 5
  batch_size: 64
  validation_split: 0.2
  callbacks:
    - type: "MetricsTracker"
    - type: "EarlyStopping"
      metric: "val_loss"
      patience: 3
    - type: "MLflowCallback"
      experiment_name: "experiment_1"
```

## Configuration System

### Main Configuration (config.yaml)

```yaml
model:
  type: "Network"
  layers:
    - name: "input_layer"
      input_size: 500
      output_size: 1
      kernel:
        type: "DENKernel"
        n: 500
        tau_s: 15
        tau_m: 3.75

optimizer:
  type: MomentumOptimizer
  lr: 100
  momentum: 0.99

lr_scheduler:
  type: StepLR
  args:
    step_size: 50
    gamma: 0.9

dataset:
  type: "RandomDataset"
  input_size: 500
  len: 1000
  encoder:
    type: "LatencyEncoder"
    size: 500
```

### Environment Configuration (env.yaml)

```yaml
T: 500          # Simulation duration
dt: 1.0         # Time step
v_th: 1.0       # Voltage threshold
v_reset: 0.0    # Reset voltage
v_0: 2.12       # Initial voltage
device: "cpu"   # Computing device
```

## Setup and Usage

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd LIF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments

1. Create an experiment configuration:
   - Define model architecture in `config.yaml`
   - Set environment parameters in `env.yaml`
   - Configure experiment settings in `experiment.yaml`
   - Define trials in `trials.yaml`

2. Place configurations in experiment directory:
```
outputs/
└── experiment_name/
    ├── config/
    │   ├── config.yaml
    │   ├── env.yaml
    │   ├── experiment.yaml
    │   └── trials.yaml
    └── results/
```

3. Run the experiment:
```bash
python main.py
```

## Monitoring and Analysis

- **MLflow Integration**: Access experiment tracking UI:
  ```bash
  mlflow ui --backend-store-uri ./outputs/mlruns
  ```

- **Metrics**: Monitor training progress through:
  - Loss curves
  - Accuracy metrics
  - Spike patterns
  - Network responses

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.