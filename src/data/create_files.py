# Function to create folder splits with a fixed seed
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import src.utils.util_general as util_general
import src.engine.utils.io_utils as io_utils

# Seed Everything
seed = 42
util_general.seed_all(seed)  # set fixed seed to all libraries

# Parameters
dataset_name = 'Pelvis_2.1'
num_patients_dataset = 375
num_patient_jobs = np.arange(25, num_patients_dataset+25, step=25)[1:] # 400 maximum number of patients in the dataset + step
train_split = 0.7
val_split = 0.2
test_split = 0.1

# Files and Directories
data_dir = os.path.join("./data/interim/", dataset_name)

# Load DB
info_data = pd.read_excel(os.path.join(data_dir, "info_Pelvis_2.1.xlsx"))
if 'label' in info_data.keys(): # Create a label column associated to every cancer type class
    pass
else:
    classes = ['B', 'G', 'L', 'P', 'R']
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    info_data['label'] = pd.Series([class_to_idx[x] for x in info_data['cancer_type']])
    info_data.to_excel(os.path.join(data_dir, f'info_{dataset_name}.xlsx'), index=False)


for patient_job in num_patient_jobs:
    print(f"Patient job: {patient_job}")
    basename = f"{dataset_name}-num-{patient_job:d}"
    dest_dir_jobs = os.path.join(data_dir, "jobs", basename)
    util_general.create_dir(dest_dir_jobs)

    # Random pick max_patient patients guaranteeing the priori distribution on each class set
    if num_patients_dataset - patient_job != 0:
        sample_patients, _ = train_test_split(info_data, test_size=num_patients_dataset - patient_job, stratify=info_data['label'], random_state=patient_job)
        sample_patients.to_excel(os.path.join(dest_dir_jobs, f'{basename}.xlsx'), index=False)
    else:
        sample_patients = info_data

    # Training, test, val split
    basename += f"_train-{train_split:0.2f}_val-{val_split:0.2f}_test-{test_split:0.2f}"
    train, test = train_test_split(sample_patients, test_size=test_split, stratify=sample_patients['label'], random_state=patient_job)  # with stratify equal to y_label we mantain the prior distributin on each set
    train, val = train_test_split(train, test_size=val_split, stratify=train['label'], random_state=patient_job)

    # Save the training/validation/test split
    s = {"sample_patients": sample_patients['filename'].to_list(), "train": train['filename'].to_list(), "val": val['filename'].to_list(), "test": test['filename'].to_list()}
    with open(os.path.join(dest_dir_jobs, f"{basename}.json"), 'w') as f:
        json.dump(s, f, ensure_ascii=False, indent=4)  # save as json
    io_utils.write_pickle(s,  os.path.join(dest_dir_jobs, f"{basename}.pickle"))  # save as pickle

print("May be the force with you")