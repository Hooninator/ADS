from collections import defaultdict

import pandas as pd

import os
import time

path_prefix="$HOME/CombBLAS/tuning-experiments/samples-gnn/"


def load_spgemm2d_data(filename="samples.txt"):
    
    data_dict = defaultdict(lambda:[])
    
    # Parse features in file
    with open(filename, 'r') as file:
        for line in file:
            features = line.split(" ")
            for feature in features:
                
                if feature=='\n':
                    continue
                    
                feature_name, value = feature.split(":")[0], (feature.split(":")[1])
                
                # Make sure naming is consistent
                if feature_name=='layer-merge':
                    feature_name = 'merge'
                
                try:
                    value = float(value)
                except:
                    pass
                
                data_dict[feature_name].append(value)
    
    #TODO: Biases?
    
    # Make dataframe
    df = pd.DataFrame(data_dict)
    
    def get_matname_from_path(path):
        return path.split("/")[-1].strip()
    
    # Change matrix paths to names
    df['A-name'] = df['A-name'].apply(get_matname_from_path)
    df['B-name'] = df['B-name'].apply(get_matname_from_path)
    
    
    df['summed-time'] = df['bcast-A'] + df['bcast-B'] + df['merge'] + df['local-mult'] 
    
    return df


def load_gnn_df(features, labels, f_prefix="samples-gnn-mod"):
    
    graph_list = []
    
    file_names = list(filter(lambda s: f_prefix in s, os.listdir(path_prefix)))
    
    df_dict = defaultdict(lambda: [])

    stime = time.time()
    for fname in file_names:
        print(f"Processing {fname}...")
        with open(path_prefix+fname, 'r') as file:
            for line in file:
                if "SAMPLE" in line:
                    continue
                feats = line.split(" ")
                found_feats = []
                for f in feats:
                    name, val = f.split(":")[0], f.split(":")[1]
                    name = name.strip()
                    if "problem" in name:
                        df_dict[name].append(val)
                        found_feats.append(name)
                    elif name in features+labels and name not in found_feats:
                        try:
                            df_dict[name].append(float(val))
                            found_feats.append(name)
                        except:
                            continue
                for f in features+labels+["problem"]:
                    if f not in found_feats:
                        df_dict[f].append(None)
    etime = time.time()
    df = pd.DataFrame(df_dict)
    print(f"Processsed {len(df)} samples in {etime-stime}s")

    df["bcast"] = df["bcast-A"] + df["bcast-B"]
    df["no-bcast"] = df["local-mult"] + df["merge"]
    return df 
