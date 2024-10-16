"""
running experiments
"""
import os
import ray
import numpy as np
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from common import MODEL_NS, SPIKE_NS, DATA_NS, Configuration
from experiment.trial import Trial


MODEL_ATTRIBUTES = \
{
    # MODEL PARAMETERS:
    MODEL_NS.NUM_OUTPUTS             : 1,
    MODEL_NS.NUM_INPUTS              : 500,
    MODEL_NS.EPOCHS                  : 200,
    MODEL_NS.LR                      : 5.0,
    MODEL_NS.MOMENTUM                : 0.75,
    
    # DATA PARAMETERS:
    DATA_NS.BATCH_SIZE               : 1,
    DATA_NS.DATASET_SIZE             : 640,
    DATA_NS.NUM_CLASSES              : 2,
    DATA_NS.TRAINING_PERCENTAGE      : 50,
    DATA_NS.TESTING_PERCENTAGE       : 25,
    DATA_NS.VALIDATION_PERCENTAGE    : 25,
    DATA_NS.ROOT                     : os.path.join("C:\\", "Users", "roymi", "Workspace", "LIF", "data", "data", "random"),
    
    # SPIKE PARAMETERS:
    SPIKE_NS.T                       : 500,
    SPIKE_NS.dt                      : 1.0,
    SPIKE_NS.tau: 10,
    
    SPIKE_NS.tau_m                   : 15,
    SPIKE_NS.tau_s                   : 15 / 4,
    SPIKE_NS.v_thr                   : 1,
}

def simple_tempotron_tune_hyperparameters():
    """
    test the accuracy of different values of the hyperparameters of the simple tempotron model
    """
    
    config = Configuration(MODEL_ATTRIBUTES)
    
    # tuned parameters:
    config[SPIKE_NS.tau_m] = tune.sample_from(lambda _: np.random.randint(4, 25))
    config[SPIKE_NS.tau_s] = lambda spec: spec.config[SPIKE_NS.tau_m] / 4
    
    config[MODEL_NS.LR] = tune.sample_from(lambda _: 10 ** (3 - np.random.randint(1, 8))) # tune.loguniform(1e-4, 1e-1)
    config[MODEL_NS.MOMENTUM] = tune.sample_from(lambda _: np.random.randint(1, 10) / 10)
    
    config[DATA_NS.BATCH_SIZE] = tune.choice([1, 2, 4, 8, 16, 32])
    
    repeat = 10
    
    # Trial.run(config, report=False)
    
    scheduler = ASHAScheduler(
        max_t=config[MODEL_NS.EPOCHS],
        grace_period=5,
        reduction_factor=10)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(Trial.run),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="min",
            scheduler=scheduler,
            num_samples=repeat,
        ),
        param_space=config.dict,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    best_config = best_result.config 
    print(f"Best trial config: {best_config}")
    
    loss = best_result.metrics["loss"]
    print(f"Best trial final validation loss: {loss}")
    
    acc = best_result.metrics["accuracy"]
    print(f"Best trial final validation accuracy: {acc}")
