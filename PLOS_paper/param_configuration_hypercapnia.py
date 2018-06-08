"""Create BayesCMD configuration file for PLOS simulated data."""

import pandas as pd
import json
import sys
sys.path.append('..')
from bayescmd.abc import priors_creator

param_df = pd.read_csv('../batch/scratch/pdists_BS_PLOS.csv',
                       header=None,
                       names=['Parameter', 'Dist. Type',
                              'Min', 'Max', 'Default'],
                       index_col=0)

chosen_params = ['p_tot', 'cytox_tot_tis', 'Dp_n', 'Vol_mit',
                 'R_auto', 'R_autp', 'r_t', 'L_CV0', 'k1_n', 'a_n']

prior_dict = priors_creator(param_df.loc[chosen_params, 'Default'].to_dict(),
                            0.25)
prior_dict['k_aut'] = ["uniform", [0.3, 1.1]]


config_dict = {"model_name": "BS",
               "inputs": ["P_a", "SaO2sup", "Pa_CO2"],
               "create_params": False,
               "priors": prior_dict,
               "targets": ["Vmca", "CCO", "HHb", "HbO2"],
               "zero_flag": {
                   "Vmca": False,
                   "CCO": True,
                   "HHb": True,
                   "HbO2": True
               },
               "batch_debug": False
               }

with open('../examples/configuration_files/healthy_hypercapnia_config.json',
          'w') as f:
    json.dump(config_dict, f)
