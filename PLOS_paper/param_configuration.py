"""Create BayesCMD configuration file for PLOS simulated data."""

import pandas as pd
import json

param_df = pd.read_csv('../batch/scratch/pdists_BS_PLOS.csv',
                       header=None,
                       names=['Parameter', 'Dist. Type',
                              'Min', 'Max', 'Default'],
                       index_col=0)

chosen_params = ['v_cn', 'R_autc', 'r_t', 'sigma_coll', 'a_n', 'Vol_mit',
                 'cytox_tot_tis', 'r_t', 'p_tot', 'r_0', 'n_m',  'NADHn',
                 'r_m', 'C_im']


prior_dict = {"defaults": param_df.loc[chosen_params, 'Default'].to_dict(),
              "variation": 0.2}

config_dict = {"model_name": "BS",
               "inputs": ["P_a"],
               "create_params": True,
               "priors": prior_dict,
               "targets": ["Vmca", "CCO", "TOI", "CBF"],
               "debug": False
               }

with open('simulated_parameter_config.json', 'w') as f:
    json.dump(config_dict, f)
