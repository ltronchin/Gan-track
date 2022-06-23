import pickle
import os
import collections
import argparse

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like
from sklearn.model_selection import KFold

from src.engine.utils.nilearn_custom_utils.nilearn_utils import crop_img_to
from src.engine.utils.sitk_utils import resample_to_spacing, calculate_origin_offset


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """

    :param image_files:
    :param image_shape:
    :param crop:
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return:
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        image_list.append(read_image(
            image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list


def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))

    # CUSTOMIZING START - ADDED BY Lorenzo
    old_shape = image.shape  # 512x512xd
    if len(image_shape) == 1:
        image_shape = (image_shape[0], image_shape[0], old_shape[-1])
    elif len(image_shape) == 2:
        image_shape = (image_shape[0], image_shape[1], old_shape[-1])
    # CUSTOMIZING END - ADDED BY Lorenzo

    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
        # nib.save(image, crop_path)
    if image_shape:
        # nib.save(resize(image, new_shape=image_shape, interpolation=interpolation), resize_path)
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3,
               3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def pad_or_crop(image, new_shape, mode="constant", interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.asarray((1, 1, 1))
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)

    # np.divide(image.header.get_zooms(), zoom_level)

    old_shape = image.shape
    pad_x_1 = int((new_shape[0] - old_shape[0])/2)
    pad_x_2 = new_shape[0] - pad_x_1 - old_shape[0]
    # pad_x_1 = max(pad_x_1, 0)
    # pad_x_2 = max(pad_x_2, 0)
    pad_y_1 = int((new_shape[1] - old_shape[1])/2)
    pad_y_2 = new_shape[1] - pad_y_1 - old_shape[1]
    # pad_y_1 = max(pad_y_1, 0)
    # pad_y_2 = max(pad_y_2, 0)
    pad_z_1 = int((new_shape[2] - old_shape[2])/2)
    pad_z_2 = new_shape[2] - pad_z_1 - old_shape[2]
    # pad_z_1 = max(pad_z_1, 0)
    # pad_z_2 = max(pad_z_2, 0)

    new_data = np.copy(image.get_fdata())
    constant_values = np.min(new_data)

    if pad_x_1 > 0:
        new_data = np.pad(new_data, ((pad_x_1, pad_x_2),
                                     (pad_y_1, pad_y_2),
                                     (0, 0)),
                          mode=mode,
                          constant_values=constant_values)
    else:
        new_data = new_data[-pad_x_1:old_shape[0]+pad_x_2,
                            -pad_y_1:old_shape[1]+pad_y_2,
                            0:old_shape[2]
                            ]

    if pad_z_1 > 0:
        new_data = np.pad(new_data, ((0, 0),
                                     (0, 0),
                                     (pad_z_1, pad_z_2)),
                          mode=mode,
                          constant_values=constant_values)
    else:
        new_data = new_data[0:new_shape[0],
                            0:new_shape[1],
                            -pad_z_1:old_shape[2]+pad_z_2
                            ]

    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3,
               3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine), (pad_x_1, pad_x_2,
                                                              pad_y_1, pad_y_2,
                                                              pad_z_1, pad_z_2)


def pad(image, new_shape, mode="constant", interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.asarray((1, 1, 1))
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)

    # np.divide(image.header.get_zooms(), zoom_level)

    old_shape = image.shape
    pad_x_1 = int((new_shape[0] - old_shape[0])/2)
    pad_x_2 = new_shape[0] - pad_x_1 - old_shape[0]
    pad_x_1 = max(pad_x_1, 0)
    pad_x_2 = max(pad_x_2, 0)
    pad_y_1 = int((new_shape[1] - old_shape[1])/2)
    pad_y_2 = new_shape[1] - pad_y_1 - old_shape[1]
    pad_y_1 = max(pad_y_1, 0)
    pad_y_2 = max(pad_y_2, 0)
    pad_z_1 = int((new_shape[2] - old_shape[2])/2)
    pad_z_2 = new_shape[2] - pad_z_1 - old_shape[2]
    pad_z_1 = max(pad_z_1, 0)
    pad_z_2 = max(pad_z_2, 0)

    new_data = np.copy(image.get_fdata())
    constant_values = np.min(new_data)
    new_data = np.pad(new_data, ((pad_x_1, pad_x_2),
                                 (pad_y_1, pad_y_2),
                                 (pad_z_1, pad_z_2)),
                      mode=mode,
                      constant_values=constant_values)

    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3,
               3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine), (pad_x_1, pad_x_2,
                                                              pad_y_1, pad_y_2,
                                                              pad_z_1, pad_z_2)


def crop(image, padding, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    shape = image.shape

    zoom_level = np.asarray((1, 1, 1))
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)

    old_shape = image.shape
    pad_x_1 = padding[0]
    pad_x_2 = padding[1]
    pad_y_1 = padding[2]
    pad_y_2 = padding[3]
    pad_z_1 = padding[4]
    pad_z_2 = padding[5]

    new_data = np.copy(image.get_fdata())
    new_data = new_data[pad_x_1:shape[0]-pad_x_2,
                        pad_y_1:shape[1]-pad_y_2,
                        pad_z_1:shape[2]-pad_z_2
                        ]

    # new_spacing = new_data.shape

    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3,
               3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_nib(volume, path, affine):
    volume_temp = nib.Nifti1Image(volume, affine=affine)
    nib.save(volume_temp, path)


def split_list_kfold_for_cross_validation(ids, n_splits=5, random_state=1988):
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_list = list()
    for counter, fold_ids in enumerate(skf.split(ids, ids)):
        fold_list.append(fold_ids)
    return fold_list


def convert_patch_shape_string_2_number(patch_shape):
    def multiplyList(myList):
        # Multiply elements one by one
        result = 1
        for x in myList:
            result = result * x
        return result
    ps = [int(s) for s in patch_shape.split("-") if s.isdigit()]
    return ps, multiplyList(ps)


def main():

    print(convert_patch_shape_string_2_number("256-128-13"))


if __name__ == "__main__":
    main()
