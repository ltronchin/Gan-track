# Function to create folder splits with a fixed seed
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.models.stylegan3.genlib.utils import util_general
import src.engine.utils.io_utils as io_utils

def format_label_path_claro(list_path, current_label):
    # {
    #     "labels": [
    #         ["00000/img00000000.png", 6],
    #         ["00000/img00000001.png", 9],
    #         ... repeated for every image in the dataset
    #         ["00049/img00049999.png", 1]
    #      ]
    # }

    id_patient = list_path[0]

    id_slice = list_path[-1].split(sep='_')[-1]
    id_slice = f"{int(id_slice):05d}"
    fformat = '.pickle'

    fpath = id_patient + '/' + id_patient + '_' + id_slice + fformat
    print(f'Final label path: {fpath}')

    return [fpath, current_label]

def generate_label_files(sample_patients, split=None, dest_dir=None):

    dataset_json = {
        'labels': [
            format_label_path_claro(list_path=util_general.split_dos_path_into_components(p), current_label=l) for p, l in zip(sample_patients['img'], sample_patients['label'])
        ]
    }
    if dest_dir is None:
        return dataset_json
    else:
        with open(os.path.join(dest_dir, f"dataset_{split}.json"), 'w') as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=4)  # save as json

def main(
        dataset_name,
        train_split,
        val_split,
        test_split,
        num_patients_dataset,
        validation_method,
        n_exp,
        seed = 42,
):
    # Seed Everything
    util_general.seed_all(seed)  # set fixed seed to all libraries
    # Files and Directories
    data_dir = os.path.join("./data/interim/", dataset_name)

    if dataset_name == 'Pelvis_2.1':

        assert validation_method == 'hold_out'
        assert n_exp == 1
        assert num_patients_dataset is not None

        num_patient_jobs = np.arange(25, num_patients_dataset + 25, step=25)[1:]  # 400 maximum number of patients in the dataset + step
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
            basename += f"_val-{validation_method}_exps-{n_exp:d}_fold-{0:d}_train-{train_split:0.2f}_val-{val_split:0.2f}_test-{test_split:0.2f}"
            train, test = train_test_split(sample_patients, test_size=test_split, stratify=sample_patients['label'], random_state=patient_job)  # with stratify equal to y_label we mantain the prior distributin on each set
            train, val = train_test_split(train, test_size=val_split, stratify=train['label'], random_state=patient_job)

            print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

            # Save the training/validation/test split
            s = {"sample_patients": sample_patients['filename'].to_list(), "train": train['filename'].to_list(), "val": val['filename'].to_list(), "test": test['filename'].to_list()}
            with open(os.path.join(dest_dir_jobs, f"{basename}.json"), 'w') as f:
                json.dump(s, f, ensure_ascii=False, indent=4)  # save as json
            io_utils.write_pickle(s,  os.path.join(dest_dir_jobs, f"{basename}.pickle"))  # save as pickle

    elif dataset_name == 'claro':

        # Upload patients list along with label.
        sample_patients = pd.read_csv(os.path.join(data_dir, 'bootstrap', 'folds', 'all.txt'), sep=" ")
        # Label.
        # Generate a dataset.json for label.
        dataset_json = generate_label_files(sample_patients)

        sample_patients = np.unique([util_general.split_dos_path_into_components(p)[0] for p in sample_patients['img']]) # Select only the patient id.
        patient_job = len(sample_patients)
        print()
        print(f"Patient job: {patient_job}")
        basename_root = f"{dataset_name}-num-{patient_job:d}"
        dest_dir_jobs = os.path.join(data_dir, "jobs", basename_root)
        util_general.create_dir(dest_dir_jobs)

        # Save label
        with open(os.path.join(dest_dir_jobs, f"dataset.json"), 'w') as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=4)  # save as json

        for fold in range(n_exp):
            basename = basename_root + f"_val-{validation_method}_exps-{n_exp:d}_fold-{fold:d}_train-{train_split:0.2f}_val-{val_split:0.2f}_test-{test_split:0.2f}"
            # Jobs Generation.
            train = pd.read_csv(os.path.join(data_dir, 'bootstrap', 'folds', str(n_exp), str(fold), 'train.txt'), sep=" ")
            val =   pd.read_csv(os.path.join(data_dir, 'bootstrap', 'folds', str(n_exp), str(fold), 'val.txt'), sep=" ")
            test =  pd.read_csv(os.path.join(data_dir, 'bootstrap', 'folds', str(n_exp), str(fold), 'test.txt'), sep=" ")

            # Label.
            generate_label_files(train, f'fold-{fold}_train', dest_dir_jobs)
            generate_label_files(val,   f'fold-{fold}_val',   dest_dir_jobs)
            generate_label_files(test,  f'fold-{fold}_test',  dest_dir_jobs)

            train = np.unique([util_general.split_dos_path_into_components(p)[0] for p in train['img']])
            val = np.unique([util_general.split_dos_path_into_components(p)[0] for p in val['img']])
            test = np.unique([util_general.split_dos_path_into_components(p)[0] for p in test['img']])
            print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

            # Save the training/validation/test split
            s = {"sample_patients":list(sample_patients), "train":list(train), "val": list(val), "test": list(test)}
            with open(os.path.join(dest_dir_jobs, f"{basename}.json"), 'w') as f:
                json.dump(s, f, ensure_ascii=False, indent=4)  # save as json
            io_utils.write_pickle(s,  os.path.join(dest_dir_jobs, f"{basename}.pickle"))  # save as pickle

    print("May be the force with you")

if __name__ == "__main__":

    # Parameters
    """
    # Pelvis 2.1
    dataset_name = 'Pelvis_2.1'
    num_patients_dataset = 375
    train_split = 0.7
    val_split = 0.2
    test_split = 0.1
    num_patients_dataset=375,
    validation_method = 'hold_out'
    n_exp = 1
    
    # claro
    dataset_name = 'claro'
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    num_patients_dataset=None,
    validation_method='bootstrap',
    n_exp=5
    """

    main(
        dataset_name = 'claro',
        train_split = 0.8,
        val_split = 0.1,
        test_split = 0.1,
        num_patients_dataset=None,
        validation_method='bootstrap',
        n_exp=5
    )