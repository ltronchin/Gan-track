#!/bin/bash

python3 ./src/data/dataset_tool_mi.py --dataset Pelvis_2.1 --configuration_file ./configs/pelvis_preprocessing.yaml --data_dir ./data/data_raw/Pelvis_2.1 --interim_dir /home/lorenzo/data_m2/data/interim --reports_dir ./reports/ --resolution 256 --processing_step process_dicom_2_nifti
sleep5m

python3 ./src/data/dataset_tool_mi.py --dataset Pelvis_2.1 --configuration_file ./configs/pelvis_preprocessing.yaml --data_dir /home/lorenzo/data_m2/data/interim/Pelvis_2.1/nifti_volumes --interim_dir /home/lorenzo/data_m2/data/interim --reports_dir ./reports/ --resolution 256 --processing_step process_nifti_resized
sleep5m

python3 ./src/data/dataset_tool_mi.py --dataset Pelvis_2.1 --configuration_file ./configs/pelvis_preprocessing.yaml --data_dir /home/lorenzo/data_m2/data/interim/Pelvis_2.1/nifti_volumes_256x256 --interim_dir /home/lorenzo/data_m2/data/interim --reports_dir ./reports/ --resolution 256 --processing_step process_nifti_normalized
sleep5m

python3 ./src/data/dataset_tool_mi.py --dataset Pelvis_2.1 --configuration_file ./configs/pelvis_preprocessing.yaml --data_dir /home/lorenzo/data_m2/data/interim/Pelvis_2.1/nifti_volumes_256x256_normalized --interim_dir /home/lorenzo/data_m2/data/interim --reports_dir ./reports/ --resolution 256 --pop_range 10 --processing_step snap_pickle
sleep5m

python3 ./src/data/dataset_tool_mi.py --dataset Pelvis_2.1 --configuration_file ./configs/pelvis_preprocessing.yaml --data_dir /home/lorenzo/data_m2/data/interim/Pelvis_2.1/temp --interim_dir /home/lorenzo/data_m2/data/interim --reports_dir ./reports/ --resolution 256 --max_patients 100000 --processing_step snap_zip