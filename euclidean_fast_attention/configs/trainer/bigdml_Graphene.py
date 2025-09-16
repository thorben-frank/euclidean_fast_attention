"""The default configuration for the experiments."""

import ml_collections

from ase import units
from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.datafile = config_dict.placeholder(str)
    config.num_train = 200
    config.num_valid = 50
    config.split_seed = 0
    config.model_seed = 0
    config.max_num_nodes = 4 * 50 + 1
    config.max_num_edges = 4 * 50 * 25 + 1
    config.max_num_graphs = 4 + 1
    config.num_epochs = None
    config.num_train_steps = 250_000
    config.save_interval_steps = 5000
    config.log_loss_every_steps = 500
    config.energy_unit = units.eV
    config.length_unit = units.Angstrom
    config.pbc_bool = True
    config.auto_eval = True
    config.energy_weight = 0.01
    config.forces_weight = 0.99
    config.subtract_energy_mean = True
    config.neighbor_list_cutoff = None

    return config
