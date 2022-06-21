#!/bin/bash
python3  --dataset=Pelvis_2.1 --configuration_file=./configs/pelvis_preprocessing.yaml --data_dir=./data/data_raw --interim_dir=./data/interim --reports_dir=./reports/ --resolution 256 --processing_step=process_dicom_2_nifti
sleep 5m
python3  --dataset=Pelvis_2.1 --configuration_file=./configs/pelvis_preprocessing.yaml --data_dir=./data/data_raw --interim_dir=./data/interim --reports_dir=./reports/ --resolution 256 --processing_step=process_nifti_resized
sleep 5m
python3  --dataset=Pelvis_2.1 --configuration_file=./configs/pelvis_preprocessing.yaml --data_dir=./data/data_raw --interim_dir=./data/interim --reports_dir=./reports/ --resolution 256 --processing_step=process_nifti_normalized
sleep 5m
python3  --dataset=Pelvis_2.1 --configuration_file=./configs/pelvis_preprocessing.yaml --data_dir=./data/data_raw --interim_dir=./data/interim --reports_dir=./reports/ --resolution 256 --processing_step=snap_pickle
sleep 5m
python3  --dataset=Pelvis_2.1 --configuration_file=./configs/pelvis_preprocessing.yaml --data_dir=./data/data_raw --interim_dir=./data/interim --reports_dir=./reports/ --resolution 256 --processing_step=snap_zip
