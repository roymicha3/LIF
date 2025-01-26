# LIF Project

This project implements a Leaky Integrate-and-Fire (LIF) model for neural simulations. It includes modules for configuring the model, running experiments, and visualizing results.

## Project Structure

- `main.py`: The main script to run the project.
- `common/`: Contains common utilities and configurations.
- `analysis/`: Contains modules for data analysis and visualization.
- `experiment/`: Contains modules for running experiments.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/LIF.git
    cd LIF
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the main script, execute:
```sh
python main.py
```

## Configuration

The model attributes can be configured in the `main.py` file. Here is an example configuration:
```python
MODEL_ATTRIBUTES = {
    MODEL_NS.NUM_OUTPUTS: 1,
    MODEL_NS.NUM_INPUTS: 500,
    MODEL_NS.LR: 100.0,
    MODEL_NS.MOMENTUM: 0.99,
    MODEL_NS.EPOCHS: 1000,
    MODEL_NS.BETA: 50,
    DATA_NS.BATCH_SIZE: 64,
    DATA_NS.DATASET_SIZE: 1000,
    DATA_NS.NUM_CLASSES: 2,
    DATA_NS.TRAINING_PERCENTAGE: 50,
    DATA_NS.TESTING_PERCENTAGE: 25,
    DATA_NS.VALIDATION_PERCENTAGE: 100,
    DATA_NS.ROOT: os.path.join(".", "data", "data", "random"),
    SPIKE_NS.T: 500,
    SPIKE_NS.dt: 1.0,
    SPIKE_NS.tau: 10,
    SPIKE_NS.tau_m: 10,
    SPIKE_NS.tau_s: 10 / 4,
    SPIKE_NS.v_thr: 1,
}
```

## Running Experiments

To run an experiment, modify the `main.py` file to call the desired experiment function. For example:
```python
from experiment.experiment import simple_tempotron_tune_hyperparameters

def main():
    config = Configuration(MODEL_ATTRIBUTES)
    simple_tempotron_tune_hyperparameters()
```

## Visualization

To visualize the results, use the visualization modules in the `analysis` package. For example:
```python
from analysis.results import RandomSpikePattern

def main():
    config = Configuration(MODEL_ATTRIBUTES)
    visualizer = RandomSpikePattern(config)
    visualizer.den_response()
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.