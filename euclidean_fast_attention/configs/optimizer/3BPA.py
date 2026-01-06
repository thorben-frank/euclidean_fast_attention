"""The default configuration for the experiments."""

import ml_collections
from ml_collections import config_dict

def get_config():
    """Get the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    config.name = "adam"
    config.learning_rate = 1e-3
    config.schedule = "exponential_decay"
    config.stop_learning_rate = 1e-5
    config.clip_by_global_norm = 1.5
    config.eps = config_dict.placeholder(float)

    return config
