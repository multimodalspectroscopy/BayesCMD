"""Create BayesCMD configuration file for PLOS simulated data."""

import pandas as pd
import json
import sys
sys.path.append('..')
from bayescmd.abc import priors_creator

param_df = pd.read_csv('../batch/scratch/pdists_BS_PLOS_wide.csv',
                       header=None,
                       names=['Parameter', 'Dist. Type',
                              'Min', 'Max', 'Default'],
                       index_col=0)

chosen_params = ['sigma_coll',
                 'R_auto',
                 'n_h',
                 'r_t',
                 'mu_max',
                 'n_m',
                 'r_m',
                 'P_v',
                 'phi',
                 'Xtot']

prior_dict = priors_creator(param_df.loc[chosen_params, 'Default'].to_dict(),
                            0.5)



config_dict = {"model_name": "BS",
               "inputs": ["SaO2sup"],
               "create_params": False,
               "priors": prior_dict,
               "targets": ["TOI", "CCO", "HHb", "HbO2"],
               "zero_flag": {
                   "TOI": False,
                   "CCO": True,
                   "HHb": True,
                   "HbO2": True
               },
               "batch_debug": False
               }

with open('../examples/configuration_files/healthy_hypoxia_config.json',
          'w') as f:
    json.dump(config_dict, f)
