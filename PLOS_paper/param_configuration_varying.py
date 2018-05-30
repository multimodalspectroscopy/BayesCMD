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
chosen_params = ['a_n', 'r_t', 'r_m', 'phi', 'Xtot', 'sigma_coll', 'v_pn',
                 'cytox_tot_tis', 'Vol_mit']

prior_dict = priors_creator(param_df.loc[chosen_params, 'Default'].to_dict(),
                            0.5)
prior_dict['k_aut'] = ["uniform", [0.3, 1.1]]


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
