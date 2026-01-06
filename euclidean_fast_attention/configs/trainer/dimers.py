"""The default configuration for the experiments."""

import ml_collections

from ase import units
from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.datafile = config_dict.placeholder(str)
    config.num_train = 4_250
    config.num_valid = 250
    config.split_seed = 0
    config.model_seed = 0
    config.max_num_nodes = 16 * 19 + 1
    config.max_num_edges = 16 * 16 * 19 + 1
    config.max_num_graphs = 16 + 1
    config.num_epochs = 2_000
    config.num_train_steps = None
    config.save_interval_steps = 2_000
    config.log_loss_every_steps = 500
    config.energy_unit = units.eV
    config.length_unit = units.Angstrom
    config.pbc_bool = False
    config.auto_eval = True
    config.energy_weight = 1.0
    config.forces_weight = 1.0
    config.subtract_energy_mean = False
    config.neighbor_list_cutoff = None

    return config
