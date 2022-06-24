# Function to create folder splits with a fixed seed
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import collections
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

import src.utils.util_general as util_general

# Seed Everything
seed = 0
util_general.seed_all(seed)  # set fixed seed to all libraries

# todo upload all the patients in a list

data_dir = '/home/lorenzo/data_m2/data/interim/Pelvis_2.1/temp'

# Parameters
y_label = "Prognosis"
cv = 10
test_size = 0.
val_size = 0.1  # 10% of train set

# Files and Directories
data_dir = "../data/AIforCOVID"
dest_dir = "./data/processed/folds"
clinical_data_path = os.path.join(data_dir, "AIforCOVID.xlsx")  # upload clinical data only to extract the patients' label
dest_dir_folds = os.path.join(dest_dir, str(cv))
util_general.create_dir(dest_dir)

# Load DB
clinical_data = pd.read_excel(clinical_data_path, index_col="ImageFile")

# K-Folds CV
fold_data = collections.defaultdict(lambda: {})
for fold in range(cv):
    train, test = train_test_split(clinical_data, test_size=test_size, stratify=clinical_data[y_label], random_state=fold)  # with stratify equal to y_label we mantain the prior distributin on each set
    train, val = train_test_split(train, test_size=val_size, stratify=train[y_label], random_state=fold)
    fold_data[fold]['train'] = train.index.to_list()
    fold_data[fold]['val'] = val.index.to_list()
    fold_data[fold]['test'] = test.index.to_list()

# all.txt
with open(os.path.join(dest_dir, 'all.txt'), 'w') as file:
    file.write("id label\n")
    for id_patient in clinical_data.index:
        label = "%s\n" % clinical_data.loc[id_patient, y_label]
        row = "%s %s" % (id_patient, label)
        file.write(row)

# create split dir
steps = ['train', 'val', 'test']
for fold in range(cv):
    dest_dir_cv = os.path.join(dest_dir_folds, str(fold))
    util_general.create_dir(dest_dir_cv)

    # .txt
    for step in steps:
        with open(os.path.join(dest_dir_cv, '%s.txt' % step), 'w') as file:
            file.write("id label\n")
            for id_patient in tqdm(fold_data[fold][step]):
                label = "%s\n" % clinical_data.loc[id_patient, y_label]
                row = "%s %s" % (id_patient, label)
                file.write(row)