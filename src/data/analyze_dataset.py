"""
Created on Mon Dec 03
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

#  Libraries
print('Import the library')
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import glob
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat


from src.engine.utils.path_utils import split_dos_path_into_components
from src.engine.utils.print_utils import print_processing, print_separator
from src.engine.utils.volume import get_size
from src.engine.utils.volume import get_max_min_intensity
from src.engine.utils.volume import get_unique_label
from src.engine.utils.volume import get_shape
from src.engine.utils.volume import get_spacing, get_percentile


def get_header_info(path):
    folders = split_dos_path_into_components(path)
    N = len(folders)
    dataset = folders[N - 4]
    folder = folders[N - 3]
    name = folders[N - 2]
    modality = folders[N - 1].replace(".nii.gz", "")
    return dataset, folder, name, modality


def generate_columns(subject_dirs, columns, dataset="ibsr"):
    subject_dir = subject_dirs[0]
    volume = nib.load(subject_dir)
    volume = volume.get_fdata()
    n_labels = len(get_unique_label(volume))

    if dataset == "brats19":
        n_labels = 4

    for label in range(1, n_labels, 1):
        columns.append("label_{}_depth".format(str(label)))
        columns.append("label_{}_percentage".format(str(label)))

    return columns


def analyze_one_file(index, subject_dirs): # analyze one modality.
    subject_dir = subject_dirs[index]
    print_processing(subject_dir)

    dataset, folder, name, modality = get_header_info(subject_dir)

    volume = nib.load(subject_dir)
    spacing = get_spacing(volume)
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    volume = volume.get_fdata()
    size = get_size(subject_dir)
    shape = get_shape(volume)
    shape_x, shape_y, shape_z = (
        shape[0],
        shape[1],
        shape[2],
    )
    (
        max_intensity,
        min_intensity,
        min_intensity_non_zeros,
    ) = get_max_min_intensity(volume)

    percentile_99_5, percentile_00_5 = get_percentile(volume)

    return (
        spacing,
        spacing_x,
        spacing_y,
        spacing_z,
        dataset,
        folder,
        name,
        modality,
        size,
        shape,
        shape_x,
        shape_y,
        shape_z,
        max_intensity,
        min_intensity,
        min_intensity_non_zeros,
        percentile_99_5,
        percentile_00_5,
    )


def analyze_one_folder(folder, dataset, output_dir, overwrite=False):
    analysis_dir = output_dir
    analysis_file_path = os.path.join(
        analysis_dir, f"{dataset}_structure_analysis.xlsx"
    )
    print("save to dir", analysis_dir)
    print("save to file", analysis_file_path)

    if not os.path.exists(analysis_dir):
        print_separator()
        print("making dir", analysis_dir)
        os.makedirs(analysis_dir)

    if overwrite or not os.path.exists(analysis_file_path):
        writer = pd.ExcelWriter(analysis_file_path)

        subject_dirs = glob.glob(os.path.join(folder, "*", "*"))

        index = range(0, len(subject_dirs) - 1, 1)

        columns = [
            "dataset",
            "folder",
            "name",
            "modality",
            "size",
            "spacing",
            "spacing_x",
            "spacing_y",
            "spacing_z",
            "shape",
            "shape_x",
            "shape_y",
            "shape_z",
            "max_intensity",
            "min_intensity",
            "min_intensity_non_zeros",
            "percentile_99_5",
            "percentile_00_5",
        ]

        list_spacing, list_spacing_x, list_spacing_y, list_spacing_z, list_dataset = (
            list(),
            list(),
            list(),
            list(),
            list(),
        )

        (
            list_folder,
            list_name,
            list_modality,
            list_size,
            list_shape,
            list_shape_x,
            list_shape_y,
            list_shape_z,
        ) = (list(), list(), list(), list(), list(), list(), list(), list())

        (
            list_max_intensity,
            list_min_intensity,
            list_min_intensity_non_zeros,
            list_percentile_99_5,
            list_percentile_00_5,
        ) = (
            list(),
            list(),
            list(),
            list(),
            list(),
        )

        pool = Pool()

        l = pool.starmap(
            analyze_one_file,
            zip(
                range(len(subject_dirs)),
                repeat(subject_dirs),
            ),
        )
        pool.close()

        for i in range(len(subject_dirs)):
            (
                spacing,
                spacing_x,
                spacing_y,
                spacing_z,
                dataset,
                folder,
                name,
                modality,
                size,
                shape,
                shape_x,
                shape_y,
                shape_z,
                max_intensity,
                min_intensity,
                min_intensity_non_zeros,
                percentile_99_5,
                percentile_00_5,
            ) = l[i]

            list_spacing.append(spacing)
            list_spacing_x.append(spacing_x)
            list_spacing_y.append(spacing_y)
            list_spacing_z.append(spacing_z)
            list_dataset.append(dataset)
            list_folder.append(folder)
            list_name.append(name)
            list_modality.append(modality)
            list_size.append(size)
            list_shape.append(shape)
            list_shape_x.append(shape_x)
            list_shape_y.append(shape_y)
            list_shape_z.append(shape_z)
            list_max_intensity.append(max_intensity)
            list_min_intensity.append(min_intensity)
            list_min_intensity_non_zeros.append(min_intensity_non_zeros)
            list_percentile_99_5.append(percentile_99_5)
            list_percentile_00_5.append(percentile_00_5)

        df = pd.DataFrame(
            list(
                zip(
                    list_dataset,
                    list_folder,
                    list_name,
                    list_modality,
                    list_size,
                    list_spacing,
                    list_spacing_x,
                    list_spacing_y,
                    list_spacing_z,
                    list_shape,
                    list_shape_x,
                    list_shape_y,
                    list_shape_z,
                    list_max_intensity,
                    list_min_intensity,
                    list_min_intensity_non_zeros,
                    list_percentile_99_5,
                    list_percentile_00_5,
                )
            ),
            columns=columns,
        )
        df.to_excel(writer, "Sheet1")
        writer.save()


def main():
    # for c in ["B", "G", "L", "P", "R"]:
    #     folder = f"/mnt/SSD/github/stylegan3/database/Pelvis_2.1/preprocessed-256x256/{c}"
    #     dataset = f"Pelvis_2.1-{c}"
    #     output_dir = "/mnt/SSD/github/stylegan3/database/Pelvis_2.1/preprocessed-256x256"
    #     analyze_one_folder(folder, dataset, output_dir, overwrite=True)
    
    
        folder = './data/data_raw/pelvis/' # f"/mnt/Data/github/3DUnetCNN_BRATS/projects/brats20/database/data_train/original"
        dataset = f"preprocessed-256x256"
        output_dir = './data/interim/pelvis'
        analyze_one_folder(folder, dataset, output_dir, overwrite=True)



if __name__ == "__main__":
    main()
