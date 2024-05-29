import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import statistics as stats
import os
import time
import random
import math
import json
import pickle
import subprocess

from collections import defaultdict

from data_utils import *
from problem_results import *

path_prefix = "/global/homes/j/jbellav/CombBLAS/tuning-experiments/"
cores_per_node = 128


features = ["FLOPS",
            "m-A",
            "m-B",
            "n-A",
            "n-B",
            "nnz-A",
            "nnz-B",
            "outputNnz-intermediate",
            "outputNnz-final",
            "Nodes",
            "PPN",
            "rank"
            ]
labels = [
        "bcast-A",
        "bcast-B",
        "total-time",
        "local-mult",
        "summed-time",
        "merge"
]
n_features = len(features)


@dataclass
class PlatformParams:
    inter_beta: float
    inter_alpha: float
    gamma:float

perlmutter_params = PlatformParams(23980.54, 3.9, 5.2e-9)


def eval_spgemm(args, test_df):
    
    test_df['params'] = test_df.apply(lambda row: f"{row['Nodes']}, {row['PPN']}", axis=1)
    test_df['processes'] = test_df.apply(lambda row: f"{row['Nodes']*row['PPN']}", axis=1)

    problems = test_df['problem'].unique()

    print(problems)

    if args.problem:
        problems = [f"{args.problem}.mtx{args.problem}.mtx-permuted\n"]

    results = ProblemResults()
    i = 0
    
    print(f"Evaluating {len(problems)} problems")
    for problem in problems:

        if i%10==0:
            print(f"{i}/{len(problems)} evaluated...")

        df_problem = test_df[test_df['problem']==problem]
        
        params = df_problem['params'].unique()
        
        y_pred_arr = np.zeros(shape=(len(params)))
        y_arr = []
        processes = []
        valid_params = []


        X = df_problem[features]
        y = df_problem[args.label]
        
        nodes_cmd = int(df_problem["Nodes"].max())
        permuted = 1 if "permuted" in problem else 0
        ppn_cmd = 64
        threads = 2

        mat_name = problem.split(".")[0]
        
        cmd = f"export OMP_NUM_THREADS={threads} && mpirun -n 16 ../build/Applications/autotune /pscratch/sd/j/jbellav/matrices/{mat_name}/{mat_name}.mtx /pscratch/sd/j/jbellav/matrices/{mat_name}/{mat_name}.mtx {permuted} {nodes_cmd} 0"

        print(f"Executing {cmd}...")

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            result.check_returncode()
        except:
            print(result.stderr)
            os.system(f"rm -f info-{mat_name}x{mat_name}*")
            os.system("rm -f logfile*")
            i+=1
            continue
        
        timings = {}

        filenames = list(filter(lambda s: f"info-{mat_name}x{mat_name}" in s, os.listdir(".")))

        bcast_pred_arr, local_spgemm_pred_arr, merge_pred_arr = {}, {}, {}

        for filename in filenames:
            with open(filename, 'r') as file:
                for line in file:
                    if line.find("PREDICTION INFO")!=-1:
                        line = next(file)
                        if "Params" not in line:
                            break
                        nodes, ppn = float(line.split(" ")[0].split(":")[1].split(",")[0]), float(line.split(" ")[0].split(":")[1].split(",")[1])
                        bcast_time, local_spgemm_time, merge_time = map(lambda s: s.split(":")[1], line.split(" ")[1:-1])
                        if f"{nodes}, {ppn}" in params:
                            bcast_pred_arr[f"{nodes}, {ppn}"] = float(bcast_time) 
                            local_spgemm_pred_arr[f"{nodes}, {ppn}"] = float(local_spgemm_time) 
                            merge_pred_arr[f"{nodes}, {ppn}"] = float(merge_time) 

        with open(f"info-{mat_name}x{mat_name}-0.out", 'r') as file:
            for line in file:
                if line.find("RUNTIME ESTIMATES")!=-1:
                    
                    line = next(file)
                    trials = line.split(" ")
                    
                    for trial in trials[:-1]:
                        params_curr,runtime = trial.split(":")[0], float(trial.split(":")[1][:-1])
                        nodes,ppn = float(params_curr.split(",")[0]), float(params_curr.split(",")[1])
                        if f"{nodes}, {ppn}" in params:
                            y_pred_arr[list(params).index(f"{nodes}, {ppn}")] = runtime

                if line.find("Prediction:")!=-1 and line.find("%")==-1:
                    t = float(line.split(":")[1])
                    timings["Prediction"] = t
                if line.find("FeatureInit:")!=-1 and line.find("%")==-1:
                    t = float(line.split(":")[1])
                    timings["FeatureInit"] = t
                if line.find("TuneSpGEMM2D")!=-1 and line.find("%")==-1:
                    t = float(line.split(":")[1])
                    timings["TuneSpGEMM2D"] = t
                if line.find("PredSpGEMMTime")!=-1 and line.find("%")==-1:
                    t = float(line.split(":")[1])
                    timings["PredSpGEMMTime"] = t


        os.system(f"rm -f info-{mat_name}x{mat_name}*")
        os.system("rm -f logfile*")

        y_arr = np.zeros(shape=(len(params)))
        
        for param in params:
            param_time = df_problem[df_problem["params"]==param][args.label].max()
            y_arr[list(params).index(param)] = param_time 


        print(y_pred_arr)

        results.add_result(problem, y_arr, y_pred_arr, 0.0, timings, bcast_pred_arr,
                           local_spgemm_pred_arr, merge_pred_arr, list(params))

        i+=1

        # Save once per trial, just incase we don't finish before batch job expires!
        with open(f"{args.pklname}.pkl", 'wb') as picklefile:
            pickle.dump(results, picklefile)


def correctness(df, mat_name):

    problem = f"{mat_name}.mtx{mat_name}.mtx\n"

    df_problem = df[df["problem"]==problem]

    df_problem = df_problem.groupby(by=["Nodes","PPN"])
    
    logfiles = list(filter(lambda s: "logfile" in s, os.listdir(".")))

    n_features = 11 

    for (nodes, ppn), df_params in df_problem:
        
        for logfile in logfiles:

            with open(logfile, 'r') as file:
                for line in file: 
                    if "FeatureMat" in line:
                        features_data = line.split(":")[1].split(" ")

                        ranks = len(features_data)//n_features

                        file_nodes = float(features_data[9])
                        file_ppn = float(features_data[10])

                        print(nodes, file_nodes, ppn, file_ppn)

                        if nodes!=file_nodes or ppn!=file_ppn:
                            break

                        correct = True
                        
                        for rank in range(ranks):

                            df_rank = df_params[df_params["rank"]==rank]

                            rank_features = features_data[(rank*n_features):(rank*n_features)+n_features]

                            for r in range(len(rank_features)):

                                name = features[r]
                                val = float(rank_features[r])
                                
                                print(f"{name} -> {val}:{df_rank[name].item()}")

                                if abs(val-df_rank[name].item())>=100:
                                    correct=False
                        if not correct:
                            print(f"Correctness check failed for ({nodes}, {ppn})")
                            return

    print(f"Correctness for {mat_name} passed!")

                

def split(df, size):    
    problems = df['problem'].unique()
    s = int(len(problems)*size)

    test_problems, train_problems = problems[:s],problems[s:]
    
    test, train = df[df['problem'].isin(test_problems)], df[df['problem'].isin(train_problems)]
    return test,train 



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--label",type=str)
    parser.add_argument("--problem",type=str)
    parser.add_argument("--pklname", type=str)
    parser.add_argument("--dfname", type=str, default="master-df-gnn")
    parser.add_argument('--load', const=1, nargs='?', type=int)
    parser.add_argument('--correctness', const=1, nargs='?', type=int)

    args = parser.parse_args()
    
    # Load in dataframe
    if args.load:
        df = load_gnn_df(features, labels) 

        # Only problems with all nodes
        all_problems = df['problem'].unique()
        valid_problems = [p for p in all_problems if df[df['problem']==p]['Nodes'].unique().shape[0]==7]
        df = df[df['problem'].isin(valid_problems)]

        print(f"{len(valid_problems)} total problems...")
        
        df.to_pickle(f"./tuning-dataframes/{args.dfname}")
    else:
        df = pd.read_pickle(f"./tuning-dataframes/{args.dfname}")
    
    if args.correctness:
        correctness(df, args.problem)
    else:
        eval_spgemm(args, df)
    

    
