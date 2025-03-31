# LIF Neural Network Framework

A PyTorch-based implementation of Leaky Integrate-and-Fire (LIF) spiking neural networks with extensive configuration options and experimental pipeline support.

## Project Structure

The project is organized into several key components:

### 1. Experiment System
- `experiment/` - Contains the core experimentation framework
  - `experiment.py` - Main experiment class handling configuration and execution
  - `trial.py` - Individual trial management and execution
  - `db/` - Database management for experiment tracking and results storage

### 2. Neural Network Components
- `network/` - Core neural network implementation
  - `topology/` - Network structure and connectivity
    - `network.py` - Main network class
    - `neuron.py` - LIF neuron implementation
    - `connection.py` - Synaptic connections
  - `kernel/` - Neural dynamics implementations
    - `leaky_kernel.py` - LIF neuron dynamics
    - `den_kernel.py` - Dendritic processing
  - `learning/` - Learning rules and updates
    - `integrate_lr.py` - Integrated learning rule
    - `single_spike_lr.py` - Single spike-based learning
  - `activation/` - Activation functions
  - `loss/` - Loss functions
  - `optimizer/` - Optimization algorithms

### 3. Data Processing
- `data/` - Data handling and preprocessing
  - `dataset/` - Dataset implementations
  - `spike/` - Spike data representations
- `encoders/` - Input encoding methods
  - `spike/` - Spike encoding schemes

### 4. Pipeline
- `pipeline/` - Training and evaluation pipeline
  - `callback/` - Training process monitors and controls

## Key Features

1. **Flexible Neural Models**
   - Leaky Integrate-and-Fire (LIF) neurons
   - Configurable membrane dynamics
   - Multiple connection types

2. **Learning Mechanisms**
   - Various spike-based learning rules
   - Gradient-based optimization
   - Customizable learning parameters

3. **Experiment Management**
   - Configuration-based experiment setup
   - Automatic experiment tracking
   - Result storage and analysis

4. **Data Processing**
   - Multiple spike encoding methods
   - Built-in dataset support
   - Custom data pipeline integration

## Usage

1. **Configuration**
   ```yaml
   experiment:
     name: "my_experiment"
     base_dir: "path/to/experiments"
     
   network:
     layers:
       - type: "LIFLayer"
         size: 784
       - type: "LIFLayer"
         size: 100
     
   training:
     epochs: 100
     batch_size: 32
     learning_rate: 0.001
   ```

2. **Running Experiments**
   ```python
   from experiment import Experiment
   
   # Load configuration
   experiment = Experiment("experiment_name", "base_dir", config, env_config)
   
   # Run experiment
   experiment.run()
   ```

3. **Analyzing Results**
   ```python
   from analysis import Results
   
   # Load results
   results = Results.load("experiment_path")
   
   # Generate visualizations
   results.plot_accuracy()
   results.plot_spike_patterns()
   ```

## Implementation Details

### Neural Dynamics

The LIF neurons implement the following dynamics:

1. **Membrane Potential Update**:
   ```
   V(t) = V(t-1) * exp(-dt/tau) + I(t)
   ```

2. **Spike Generation**:
   ```
   spike = V(t) >= threshold
   V(t) = V(t) * (1 - spike)
   ```

### Learning Rules

The framework supports various learning rules:

1. **Integrate Learning**:
   - Accumulates gradients over time
   - Applies updates based on integrated activity

2. **Single Spike Learning**:
   - Updates based on individual spike timing
   - Temporal window-based weight updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
