"""The default configuration for the experiments."""

import ml_collections

from ase import units
from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.datafile = config_dict.placeholder(str)
    config.num_train = 400_000
    config.num_valid = 5_000
    config.split_seed = 0
    config.model_seed = 0
    config.max_num_nodes = 32 * 6 + 1
    config.max_num_edges = 32 * 6 * 6 + 1  # consider fully connected graphs due to the dispersion module
    config.max_num_graphs = 32 + 1
    config.num_epochs = 500
    config.num_train_steps = None
    config.save_interval_steps = 5_000
    config.log_loss_every_steps = 500
    config.energy_unit = units.eV
    config.length_unit = units.Angstrom
    config.pbc_bool = False
    config.auto_eval = True
    config.energy_weight = 0.01
    config.forces_weight = 0.99
    config.subtract_energy_mean = False
    config.neighbor_list_cutoff = 100.0  # important for dispersion module, since we only have one set of neighbor list indices at the moment

    return config
