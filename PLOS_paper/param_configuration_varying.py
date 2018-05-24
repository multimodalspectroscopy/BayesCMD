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

# 25% variation
# chosen_params = ['v_cn', 'R_autc', 'a_n', 'Vol_mit',
#                  'cytox_tot_tis', 'r_t', 'p_tot', 'v_on']


## 50% variation
chosen_params = ['r_m', 'v_pn', 'sigma_coll', 'p_tot', 'r_t', 'Vol_mit',
                 'cytox_tot_tis', 'r_0', 'v_cn', 'T_max0', 'Xtot', 'Dp_n',
                 'a_n', 'K_sigma', 'R_autp', 'K_G']

prior_dict = priors_creator(param_df.loc[chosen_params, 'Default'].to_dict(),
                            0.5)
prior_dict['k_aut'] = ["uniform", [0.3, 1.0]]


config_dict = {"model_name": "BS",
               "inputs": ["P_a"],
               "create_params": False,
               "priors": prior_dict,
               "targets": ["Vmca", "CCO", "TOI"],
               "debug": False
               }

with open('../examples/configuration_files/varying_parameter_wide_config.json',
          'w') as f:
    json.dump(config_dict, f)
