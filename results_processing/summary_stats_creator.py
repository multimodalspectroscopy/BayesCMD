"""Get summary statistics and distances using existing models runs."""
import numpy as np
import pandas as pd
import json
import os
import csv
import argparse
from distutils import dir_util
from bayescmd import abc

ap = argparse.ArgumentParser('Choose results to process:')
ap.add_argument(
    'parent_dir',
    metavar="PARENT_DIR",
    help='Parent directory holding model run folders')

ap.add_argument(
    'true_data',
    metavar="TRUE_DATA",
    help='csv file containing the true data')

ap.add_argument(
    'conf',
    metavar="config_file",
    help='Config file used to generate model runs')

args = ap.parse_args()

with open(args.conf, 'r') as conf_f:
    conf = json.load(conf_f)


data_dirs = [os.path.join(args.parent_dir, d)
             for d in os.listdir(args.parent_dir)
             if os.path.isdir(os.path.join(args.parent_dir, d))]


true_data = pd.read_csv(args.true_data)
d0 = true_data.to_dict(orient='list')
true_means = true_data.mean()
true_std = true_data.agg(np.std, ddof=0)

outputs = conf['targets']


for data_directory in data_dirs:
    clean_path = os.path.join(data_directory, 'clean_data')
    dir_util.mkpath(clean_path)
    print(data_directory)
    data_runs = [f for f in os.listdir(data_directory) if f[:6] == 'output']
    data_runs.sort(key=lambda x: int(os.path.splitext(x)[0][7:]))
    # This needs work to be more general, leaving as is for now ####
    header = ['ii', 'CCO', 'DHbT', 'P_a', 'SpO2', 'temp', 't']
    # dataframes = []
    for r in data_runs:
        new_rows = []
        print(r)
        k = os.path.splitext(r)[0][7:]
        with open(os.path.join(data_directory, r), 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) == 3:
                    row.extend([np.nan, np.nan, np.nan, np.nan])
                elif len(row) == 7:
                    ii = row.pop()
                    t = row.pop(0)
                    row.insert(0, ii)
                    row.append(t)
                new_rows.append(row)

        with open(os.path.join(clean_path, 'cleaned_{}.csv'.format(k)),
                  'w') as wf:
            writer = csv.writer(wf)
            writer.writerows(new_rows)

        print("Merging data")
        all_Data = pd.DataFrame(new_rows,
                                columns=header).apply(pd.to_numeric,
                                                      errors='coerce')

        # all_Data = pd.concat(dataframes)

        print("Grouping data")
        grouped = all_Data.groupby(['ii'])
        print("Calculating mean")
        sim_means = grouped.mean()
        print("Calculating STD")
        sim_std = grouped.agg(np.std, ddof=0)

        mean_distances = sim_means[outputs] - true_means[outputs]
        mean_distances.rename(
            columns={"CCO": "CCO_mean", "DHbT": "DHbT_mean"}, inplace=True)
        std_distances = sim_std[outputs] - true_std[outputs]
        std_distances.rename(
            columns={"CCO": "CCO_std", "DHbT": "DHbT_std"}, inplace=True)

        summary_distances = pd.concat([mean_distances, std_distances], axis=1)
        summary_distances['CCO_distance'] = summary_distances[[
            'CCO_mean', 'CCO_std']].sum(axis=1)
        summary_distances['DHbT_distance'] = summary_distances[[
            'DHbT_mean', 'DHbT_std']].sum(axis=1)
        summary_distances['total_distance'] = summary_distances.sum(axis=1)

        summary_distances.reset_index(drop=True, inplace=True)

        df = pd.DataFrame({"Euclidean": [],
                           "CCO_Euclidean": [],
                           "DHbT_Euclidean": []})
        for name, group in grouped:
            distances = abc.get_distance(d0,
                                         group.to_dict(orient='list'),
                                         outputs,
                                         distance='euclidean',
                                         zero_flag={
                                             k: False for k in outputs},
                                         normalise=False)
            temp = pd.DataFrame({**distances, **{'ii': name}}, index=[0])
            temp.rename(columns={"TOTAL": "Euclidean",
                                 "CCO": "CCO_Euclidean",
                                 "DHbT": "DHbT_Euclidean"}, inplace=True)
            df = df.append(temp, ignore_index=True)

        summary_distances = pd.concat([summary_distances, df], axis=1)
        summary_distances.to_csv(os.path.join(
            clean_path, 'summary_distances_{}.csv'.format(k)), index=False)
