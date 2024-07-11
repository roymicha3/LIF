# LIF

## Project structure:
- data:
    - data_sample
    - loaders:
        - data_loader
        - mnist_data_loader

    - spike:
        - spike_data
        - spike_sample

- encoders:
    - encoder
    - identity_encoder
    - spike:
        - spike_ecoder
        - spike_utils

- settings:
    - model_attributes
    - model_namespace
    - spike:
        - spike_namespace

- tools:
    - utils

- common

## steps inside the project:
This project is built based on the Tempotron SNN.
We will start by exploring the Tempotron capabilities on randomly generated spike trains:
    1. Generate and plot random spike trains.
    2. Explore the LIF response to the spike trains (plot and normalize).

## TODO: