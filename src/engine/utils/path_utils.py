# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import os
import ntpath
from src.engine.utils.print_utils import print_separator
from src.engine.utils.utils import str2bool


def find_is_augment(config):
    augment_flipud = (config["augment_flipud"],)
    augment_fliplr = (config["augment_fliplr"],)
    augment_elastic = (config["augment_elastic"],)
    augment_rotation = (config["augment_rotation"],)
    augment_shift = (config["augment_shift"],)
    augment_shear = (config["augment_shear"],)
    augment_zoom = config["augment_zoom"]
    augment = (
        augment_flipud[0]
        or augment_fliplr[0]
        or augment_elastic[0]
        or augment_rotation[0]
        or augment_shift[0]
        or augment_shear[0]
        or augment_zoom
    )
    if augment:
        is_augment = "1"
    else:
        is_augment = "0"
    return is_augment


def update_is_augment(args, config):
    if args.is_augment != "1":
        config["augment_flipud"] = False
        config["augment_elastic"] = False
        config["augment_rotation"] = False
        config["augment_shift"] = False
        config["augment_shear"] = False
        config["augment_zoom"] = False
        config["augment_rotation"] = False
        config["augment_fliplr"] = False
    return config


def get_project_dir(path, project_name):
    paths = path.split(project_name)
    return paths[0] + project_name


def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()
    return folders


def get_h5_image_path(
    brats_dir, is_normalize_mean_std=False, challenge=2018, dataset="original"
):
    if is_normalize_mean_std:
        dataset_fullname = "brats{}_{}_normalize_mean_std".format(challenge, dataset)
    else:
        dataset_fullname = "brats{}_{}_normalize_minh".format(challenge, dataset)

    save_to_dir = os.path.join(brats_dir, "database", dataset_fullname)
    return save_to_dir


def get_data_dir(brats_dir, data_folder="data_train", dataset="test"):
    return os.path.join(brats_dir, data_folder, dataset)


def get_analysis_dir(dataset_dir, data_folder):
    return os.path.join(dataset_dir, data_folder)


def get_normalize_minh_dir(brats_dir, data_folder="data_train", dataset="test"):
    return os.path.join(brats_dir, data_folder, dataset + "_minh_normalize")


def get_normalize_minh_file_path(path, dataset="test"):
    return path.replace(dataset, dataset + "_minh_normalize")


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]


def get_modality(path, ext=".nii.gz"):
    filename = get_filename(path)
    modality = filename.replace(ext, "")
    return modality


def make_dir(dir, is_printing=True):
    if not os.path.exists(dir):
        if is_printing:
            print_separator()
            print("making dir", dir)
        os.makedirs(dir)


def get_template_path(
    path,
    brats_dir,
    dataset="test",
    template_data_folder="data_train",
    template_folder="HGG/Brats18_2013_2_1",
):
    filename = get_filename(path)
    template_path = os.path.join(
        brats_dir, "database", template_data_folder, dataset, template_folder, filename
    )
    return template_path


def get_h5_training_dir(brats_dir, datatype="data"):
    return os.path.join(brats_dir, "database", datatype)


def get_core_name(args):
    if args.mode == "train":
        return "{}_{}_is-{}_crop-{}_bias-{}_denoise-{}_norm-{}_hist-{}".format(
            args.challenge,
            args.year,
            args.image_shape,
            str(args.crop),
            str(args.is_bias_correction),
            str(args.is_denoise),
            str(args.is_normalize),
            str(args.is_hist_match),
        )
    else:
        return "{}_{}_{}_is-{}_crop-{}_bias-{}_denoise-{}_norm-{}_hist-{}".format(
            args.mode,
            args.challenge,
            args.year,
            args.image_shape,
            str(args.crop),
            str(args.is_bias_correction),
            str(args.is_denoise),
            str(args.is_normalize),
            str(args.is_hist_match),
        )


def get_model_name(args):
    model_temp = args.model
    model_temp = "{}{}d".format(args.model, str(args.model_dim))
    if "tv" in args.loss and "tversky" not in args.loss:
        from decimal import Decimal

        args.loss = "{}-{}".format(
            args.loss, "{:.0E}".format(Decimal(str(args.weight_tv_to_main_loss)))
        )
    if any(ext in args.model for ext in config_dict["model_depth"]):
        return "ps-{}_{}_crf-{}_d-{}_nb-{}_loss-{}_aug-{}".format(
            args.patch_shape,
            model_temp,
            str(args.is_crf),
            str(args.depth_unet),
            str(args.n_base_filters_unet),
            args.loss,
            str(args.is_augment),
        )
    else:
        return "ps-{}_{}_crf-{}_loss-{}_aug-{}".format(
            args.patch_shape,
            model_temp,
            str(args.is_crf),
            args.loss,
            str(args.is_augment),
        )


def get_short_model_name(args):
    model_temp = args.model
    model_temp = "{}{}d".format(args.model, str(args.model_dim))
    if "unet" in args.model or "simple" in args.model or "eye" in args.model:
        return "ps-{}_{}_crf-{}_d-{}_nb-{}".format(
            args.patch_shape,
            model_temp,
            str(args.is_crf),
            str(args.depth_unet),
            str(args.n_base_filters_unet),
        )
    else:
        return "ps-{}_{}_crf-{}".format(args.patch_shape, model_temp, str(args.is_crf))


def get_short_core_name(args):
    return "{}_{}_is-{}_crop-{}_bias-{}".format(
        args.challenge,
        args.year,
        args.image_shape,
        str(args.crop),
        str(args.is_bias_correction),
    )


def get_finetune_name(args):
    short_model_name = get_short_model_name(args)
    short_core_name = get_short_core_name(args)
    return short_model_name, short_core_name


def get_model_baseline_path(folder, args):
    import glob

    short_model_name, short_core_name = get_finetune_name(args)
    print(folder, short_model_name, short_core_name)
    model_baseline_path = None
    for filename in glob.glob(folder + "/*"):
        # print(filename)
        if short_model_name in filename and short_core_name in filename:
            model_baseline_path = filename
    return model_baseline_path


def get_model_h5_filename(datatype, args, fold_number=0):
    core_name = get_core_name(args)
    model_full_name = get_model_name(args)
    if fold_number == 0:
        if str2bool(args.is_test):
            return "test_{}_{}_{}.h5".format(core_name, model_full_name, datatype)
        else:
            return "{}_{}_{}.h5".format(core_name, model_full_name, datatype)
    else:
        if str2bool(args.is_test):
            return "test_{}_{}_{}_fold-{}.h5".format(
                core_name, model_full_name, datatype, fold_number
            )
        else:
            return "{}_{}_{}_fold-{}.h5".format(
                core_name, model_full_name, datatype, fold_number
            )


def get_training_h5_filename(datatype, args):
    core_name = get_core_name(args)
    return "{}_{}.h5".format(core_name, datatype)


def get_mask_path_from_set_of_files(in_files):
    for file in in_files:
        if "mask.nii.gz" in file:
            return file


def get_shape_string(image_shape):
    shape_string = ""
    for i in range(len(image_shape) - 1):
        shape_string = shape_string + str(image_shape[i]) + "-"
    shape_string = shape_string + str(image_shape[-1])
    return shape_string


def get_shape_from_string(shape_string):
    splitted_string = shape_string.split("-")
    splitted_number = list(map(int, splitted_string))
    return tuple(splitted_number)


def get_input_shape_from_tuple(shape_tuple):
    shape_list = list()
    for i in range(len(shape_tuple)):
        if i == 0 or shape_tuple[i] != 1:
            shape_list.append(shape_tuple[i])
    return tuple(shape_list)


def get_training_h5_paths(brats_dir, args, is_finetune=False, dir_read_write="base"):

    fold_number = args.crossval

    data_dir = get_h5_training_dir(brats_dir, "data")
    make_dir(data_dir)
    trainids_dir = get_h5_training_dir(brats_dir, "train_val_test_ids")
    make_dir(trainids_dir)
    validids_dir = get_h5_training_dir(brats_dir, "train_val_test_ids")
    make_dir(validids_dir)
    testids_dir = get_h5_training_dir(brats_dir, "train_val_test_ids")
    make_dir(testids_dir)
    model_dir = get_h5_training_dir(brats_dir, "model")
    make_dir(model_dir)

    data_filename = get_training_h5_filename("data", args)
    model_filename = get_model_h5_filename("model", args, fold_number=fold_number)

    if fold_number == 0:
        if args.is_test == "1":
            trainids_filename = "test_train_ids.h5"
            validids_filename = "test_valid_ids.h5"
            testids_filename = "test_test_ids.h5"
        else:
            trainids_filename = "train_ids.h5"
            validids_filename = "valid_ids.h5"
            testids_filename = "test_ids.h5"
    else:
        if args.is_test == "1":
            trainids_filename = "test_train_ids_{}.h5".format(fold_number)
            validids_filename = "test_valid_ids_{}.h5".format(fold_number)
            testids_filename = "test_test_ids_{}.h5".format(fold_number)
        else:
            trainids_filename = "train_ids_{}.h5".format(fold_number)
            validids_filename = "valid_ids_{}.h5".format(fold_number)
            testids_filename = "test_ids_{}.h5".format(fold_number)

    data_path = os.path.join(data_dir, data_filename)
    if is_finetune:
        model_path = os.path.join(model_dir, dir_read_write, model_filename)
    else:
        model_path = os.path.join(model_dir, model_filename)
    trainids_path = os.path.join(trainids_dir, trainids_filename)
    validids_path = os.path.join(validids_dir, validids_filename)
    testids_path = os.path.join(testids_dir, testids_filename)

    return data_path, trainids_path, validids_path, testids_path, model_path


def get_save_directory_for_data(output_dir):
    # to prevent save data for all cases/folds
    case = get_filename_without_extension(output_dir)
    path = get_parent_dir(output_dir)
    model = get_filename_without_extension(path)
    path = get_parent_dir(path)
    folder = model.split("_")
    if "fold" in folder[-1]:
        folder = folder[-1]
    else:
        folder = "fold-0"

    return os.path.join(path, folder, case)


def main():
    output_dir = "/mnt/Data/3DUnetCNN_BRATS/projects/pros/database/prediction/pros_2018_is-256-256-128_crop-0_bias-0_denoise-0_norm-11_hist-0_ps-128-128-128_segnet3d_crf-0_loss-dice_xent_aug-1_model/validation_case_956"
    print(get_save_directory_for_data(output_dir))


if __name__ == "__main__":
    main()
