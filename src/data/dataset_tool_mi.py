# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tool for creating ZIP/PNG based datasets."""

import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
from tqdm import tqdm

# Lorenzo
import yaml
import matplotlib.pyplot as plt
from src.utils import util_general

# Minh
import glob
import dicom2nifti
from itertools import repeat
import nibabel as nib
from multiprocessing import Pool
import shutil
from random import shuffle

import src.engine.utils.path_utils as path_utils
import src.engine.utils.utils as utils
import src.engine.utils.volume as volume_utils
import src.engine.utils.io_utils as io_utils

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

def open_image_folder_patients(source_dir, *, max_patients: Optional[int], dataset):
    input_patients = glob.glob(os.path.join(source_dir, "*"))

    # Load labels.
    labels = {}

    max_idx = maybe_min(len(input_patients), max_patients)

    def iterate_images():
        for idx, patient in enumerate(input_patients):
            arch_fname = os.path.relpath(patient, source_dir)
            arch_fname = arch_fname.replace("\\", "/")

            # Read all modalities of one patient
            modalities = glob.glob(os.path.join(patient, "*.nii.gz"))
            name_modalities = [path_utils.get_filename(path) for path in modalities]
            name_modalities = [name_modalities[i].replace(".nii.gz", "") for i in range(len(name_modalities))] # remove the .nii.gz

            parent_dir = path_utils.get_parent_dir(modalities[0])
            patient_name = path_utils.get_filename(parent_dir)

            fdata = {}
            for name_modality in name_modalities:
                fdata[name_modality] = nib.load(os.path.join(parent_dir, f"{name_modality}.nii.gz")).get_fdata()

            depth = nib.load(modalities[0]).shape[-1] # save the number of slices/channel
            for d in range(depth):
                img = {}  # dictionary of modalities.
                for name_modality in name_modalities:
                    img[name_modality] = fdata[name_modality][:, :, d]  # based on depth index (d): 0000->0128
                yield dict(img=img,label=labels.get(arch_fname),  name=f"{patient_name:s}_{d:05d}",  folder_name=f"{patient_name:s}")

            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

def open_dataset_patient(source, *, max_patients: Optional[int], dataset):
    if os.path.isdir(source):
        return open_image_folder_patients(source, max_patients=max_patients, dataset=dataset)
    elif os.path.isfile(source):
        assert False, "unknown archive type"
    else:
        error(f"Missing input file or directory: {source}")

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --resolution=512x384
    """

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    if resolution is None: resolution = (None, None)
    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_image(image['img'])

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()] # pylint: disable=unsubscriptable-object
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

def convert_dicom_2_nifti(source: str, dest: str, modes_to_preprocess: list, save_to_folder: bool=False):
    """Function to merge the slices of each patient (dicom) to nifti volume (do ir for each modatility)"""
    patients = glob.glob(os.path.join(source, "*")) # patient folders

    # Cycle on patients
    for pat in patients:
        fname_pat = path_utils.get_filename_without_extension(pat)
        print(f"Patient: {pat}")
        output_dir = os.path.join(dest, fname_pat)
        if os.path.exists(output_dir):
            print(f"{output_dir} already exist! Skip this patient.")
            continue

        path_utils.make_dir(output_dir, is_printing=False)

        # Cycle on modatilies
        for fname_mode in modes_to_preprocess:
            print(f"Modatility: {fname_mode}")
            if os.listdir(pat) != fname_mode:
                mode = os.path.join(pat, os.listdir(pat)[0], fname_mode)
            else:
                mode = os.path.join(pat, fname_mode)

            output_file = os.path.join(output_dir, f"{fname_mode}.nii.gz")

            try:
                dicom2nifti.dicom_series_to_nifti(mode, output_file, reorient_nifti=True) # to dicom_series_to_nifti feed a directory containing the slices to be included in .nii.gz volume
            except:
                print(f"Fail to convert {mode:s}")

#----------------------------------------------------------------------------

def resize_file(folder_index, folders, dest, image_shape, interpolation='linear'):  # resize 1 patient folder [MR, CT, ...]
    """Process 1 patient"""

    # Folder: patient
    folder = folders[folder_index]
    name = path_utils.get_filename_without_extension(folder)

    # Files: modalitites
    files = glob.glob(os.path.join(folder, "*"))

    if os.path.isdir(files[0]):
        files = glob.glob(os.path.join(files[0], "*"))

    for file_mode in files:
        output_dir = os.path.join(dest, name)
        path_utils.make_dir(output_dir, is_printing=False)

        fname = path_utils.get_filename_without_extension(file_mode)
        output_file = os.path.join(output_dir, f"{fname}.gz")

        try:
            image = utils.read_image(file_mode, image_shape=image_shape, interpolation=interpolation, crop=None) # read and reshape image
            nib.save(image, output_file)
        except:
            raise IOError(f"fail to convert {file_mode:s}")

def resize_nifti_folder(source: str, dest: str, image_shape=(256, 256)):  # rescale from [512, 512, n_slices_patient] -> [res, res, n_slices_patient]


    folders = glob.glob(os.path.join(source, "*"))

    # try:
    #     pool = Pool()  # Multithreading
    #     l = pool.starmap(resize_file, zip(range(len(folders)), repeat(folders), repeat(dest), repeat(image_shape))) # todo check multiprocessing
    #     pool.close()
    # except:
    #     for idx_pat in range(len(folders)):
    #         resize_file(folder_index=idx_pat, folders=folders, dest=dest, image_shape=image_shape)
    for idx_pat in range(len(folders)):
        resize_file(folder_index=idx_pat, folders=folders, dest=dest, image_shape=image_shape)

#----------------------------------------------------------------------------

def get_normalization_range(data, data_options):

    # Upper value
    if data_options['upper_percentile'] is not None:
        upper = np.percentile(data, data_options['upper_percentile'])
    elif data_options['range']['max'] is not None:
        upper = data_options['range']['max']
    else:
        upper = data.max()

    # Lower value
    if data_options['lower_percentile'] is not None:
        lower = np.percentile(data, data_options['lower_percentile'])
    elif data_options['range']['min'] is not None:
        lower = data_options['range']['min']
    else:
        lower = data.min()

    return upper, lower

def normalize_per_dataset(data, dataset, modes_args, low=0.0, hi=255.0):
    """follow this https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py"""
    if dataset == "Pelvis_2.1":
        upper, lower = get_normalization_range(data, modes_args)
        data = np.clip(data, lower, upper)
        data = (data - lower) / (upper - lower) # map between 0 and 1
        # data = data * (hi - low) + low # put between 0 and 255
    elif dataset == 'claro':
        pass
    else:
        raise NotImplementedError(f"Normalization for {dataset} was not implemented.")
    return data

def normalize_file(folder_index, folders, dest, dataset, modes_args):
    # Function that works per patient

    folder = folders[folder_index]
    name = path_utils.get_filename_without_extension(folder)
    files = glob.glob(os.path.join(folder, "*"))

    # Get all modalitites.
    if os.path.isdir(files[0]):
        files = glob.glob(os.path.join(files[0], "*"))

    # For each modality.
    for file_mode in files:
        output_dir = os.path.join(dest, name)
        path_utils.make_dir(output_dir, is_printing=False)  # Make directory for each patient if not exist.

        fname = path_utils.get_filename_without_extension(file_mode)
        output_file = os.path.join(output_dir, f"{fname}.gz")

        try:
            print(f"Reading: {file_mode}")
            volume = nib.load(file_mode)
            data = volume.get_fdata()

            data = normalize_per_dataset(data, dataset, modes_args[fname.split('.')[0]])

            affine = volume.affine
            image = nib.Nifti1Image(data, affine=affine)
            nib.save(image, output_file)

        except:
            raise IOError(f"fail to normalize {file_mode:s}")

def normalize_folder(source: str, dest: str,  dataset: str,  modes_args: dict):
    folders = glob.glob(os.path.join(source, "*"))

    #try:
    #    pool = Pool() # Multithreading
    #    l = pool.starmap(normalize_file, zip(range(len(folders)), repeat(folders), repeat(dest), repeat(dataset)))
    #    pool.close()
    #except:
    #    for idx_pat in range(len(folders)):
    #        normalize_file(folder_index=idx_pat, folders=folders, dest=dest, dataset=dataset, modes_arg=modes_args)
    for idx_pat in range(len(folders)):
        normalize_file(folder_index=idx_pat, folders=folders, dest=dest, dataset=dataset, modes_args=modes_args)

#----------------------------------------------------------------------------

def convert_dataset_mi(
    source: str,
    dest: str,
    max_patients: Optional[int],
    dataset: str,
    is_overwrite=False,
):

    # Open normalized folder.
    num_files, input_iter = open_dataset_patient(
        source, max_patients=max_patients, dataset=dataset
    )

    # Create a temp folder to be save into zipfile
    temp = os.path.join(dest, "temp")

    if os.path.isdir(temp) and is_overwrite:
        print(f"Removing {temp}")
        shutil.rmtree(temp)
    path_utils.make_dir(temp, is_printing=False)

    dataset_attrs = None

    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        folder_name = image["folder_name"]
        idx_name = image["name"]
        archive_fname = f"{folder_name}/{idx_name}.pickle"
        path_utils.make_dir(os.path.join(temp, f"{folder_name}"), is_printing=False)
        out_path = os.path.join(temp, archive_fname)

        if not is_overwrite and os.path.exists(out_path):
            continue

        img = image["img"]
        # Sanity check
        from PIL import Image
        for m in list(img.keys()):
            sanity_check_dir = os.path.join(dest, 'sanity_check', folder_name, m)
            path_utils.make_dir(sanity_check_dir, is_printing=False)
            im = Image.fromarray(img[m])
            im.save(os.path.join(sanity_check_dir, f"{idx_name}.tif"), 'tiff', compress_level=0, optimize=False)

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across # the whole dataset
        modalities = sorted(img.keys())
        cur_image_attrs = { "width": img[modalities[0]].shape[1], "height": img[modalities[0]].shape[0], "modalities": modalities}
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs["width"]
            height = dataset_attrs["height"]
            if width != height:
                error(f"Image dimensions after scale and crop are required to be square.  Got {width}x{height}")
            if width != 2 ** int(np.floor(np.log2(width))):
                error("Image width/height after scale and crop are required to be power-of-two")
        elif dataset_attrs != cur_image_attrs:
            err = [f"  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}" for k in dataset_attrs.keys()]  # pylint: disable=unsubscriptable-object
            error(f"Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n" + "\n".join(err))

        # Save the dict as a pickle.
        io_utils.write_pickle(img, out_path)

# ----------------------------------------------------------------------------

def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def split_list_cross_validation(input_list, n_fold=5, shuffle_list=True, is_test=False):
    if shuffle_list and not is_test:
        shuffle(input_list)
    if is_test:
        print(input_list)
    n_valid_sample = round(len(input_list) / 5)
    fold_list = list()
    for i_fold in range(n_fold):
        n_start = i_fold * n_valid_sample
        if i_fold < n_fold - 1:
            n_end = (i_fold + 1) * n_valid_sample
        else:
            n_end = len(input_list)
        val_list = input_list[n_start:n_end]
        if i_fold == 0:
            train_list = [e for i, e in enumerate(input_list) if i >= n_end]
        elif i_fold == n_fold - 1:
            train_list = [e for i, e in enumerate(input_list) if i < n_start]
        else:
            train_list = [e for i, e in enumerate(input_list) if i < n_start] + [
                e for i, e in enumerate(input_list) if i >= n_end
            ]
        fold_list.append([train_list, val_list])
    if is_test:
        print(fold_list)
    return fold_list


def write_to_zip(source: str, dest=None, max_patients=30, split=None):
    if split is None:
        split = {"train": 0.8, "val": 0.2, "test": 0}

    def add_to_zip(zipObj, patient, split):
        files = glob.glob(os.path.join(patient, "*.pickle"))
        if len(files) == 0:
            files = glob.glob(os.path.join(patient, "*.png"))

        print(f">> Writing {patient} to zip file")
        for file in files:
            filename = os.path.join(
                split,
                path_utils.get_filename_without_extension(patient),
                path_utils.get_filename(file),
            )
            # Add file to zip
            zipObj.write(file, filename)

    # Get all patients in temp folder.
    patients = glob.glob(os.path.join(source, "*"))
    # Get only the names of patients
    patients = [path_utils.get_filename_without_extension(patient) for patient in patients]
    assert len(patients) > 0

    max_patients = min(max_patients, len(patients))
    train_split, val_split, test_split = split["train"], split["val"], split["test"]
    basename = f"num-{max_patients:d}_train-{train_split:0.2f}_val-{val_split:0.2f}_test-{test_split:0.2f}"

    # Load Splitted dataset if existed, if not make a new one.
    if dest is None:
        parent_dir = path_utils.get_parent_dir(source)
    else:
        parent_dir = dest

    split_path = os.path.join(parent_dir, "train_val_test_ids", f"{basename}.pickle")
    path_utils.make_dir(path_utils.get_parent_dir(split_path), is_printing=False)

    if os.path.exists(split_path):
        s = io_utils.read_pickle(split_path)
        train_patients, val_patients, test_patients = s["train"], s["val"], s["test"]
    else:
        pass

        # Shuffle and take max_patients samples from dataset.
        patients = sorted(patients)
        shuffle(patients)

        sample_patients = patients[:max_patients]
        train_patients, val_test_patients = split_list(sample_patients, train_split)
        val_patients, test_patients = split_list(val_test_patients, val_split / (val_split + test_split))

        s = {"train": train_patients, "val": val_patients, "test": test_patients}
        import json
        with open(os.path.join(parent_dir, "train_val_test_ids", f"{basename}.json"), 'w') as f:
            json.dump(s, f, ensure_ascii=False, indent=4) # save as json

        io_utils.write_pickle(s, split_path) # save as pickle

    # Init zip file.
    out_path = os.path.join(parent_dir, f"{basename}.zip",)

    # Write to zip
    with zipfile.ZipFile(out_path, "w") as zipObj:
        for patient in train_patients:
            patient_path = os.path.join(source, patient)
            add_to_zip(zipObj, patient_path, "train")
        for patient in val_patients:
            patient_path = os.path.join(source, patient)
            add_to_zip(zipObj, patient_path, "val")
        for patient in test_patients:
            patient_path = os.path.join(source, patient)
            add_to_zip(zipObj, patient_path, "test")

# ----------------------------------------------------------------------------

def prepare_Pelvis_2_1(dataset,
                       resolution,
                       source_dir,
                       dest_dir,
                       modes_args,
                       validation_args,
                       process_dicom_2_nifti=None,
                       process_nifti_resized=None,
                       process_nifti_normalized=None,
                       snap_pickle=None,
                       snap_zip=None):

    # From dicom to nifti
    dest_dir_nifti = os.path.join(dest_dir, 'nifti_volumes')
    if process_dicom_2_nifti is not None:
        print(f"Convert to nifti, output folder: {dest_dir_nifti}")
        convert_dicom_2_nifti(source=source_dir, dest=dest_dir_nifti, modes_to_preprocess=list(modes_args.keys()))

    # Resize nifti volume from [original_res, original_res, n_slices] to [res x res x n_slices]
    dest_dir_nifti_resized = os.path.join(dest_dir, f'nifti_volumes_{resolution}x{resolution}')
    if process_nifti_resized is not None:
        print(f"Resize to resolution {resolution}, output folder: {dest_dir_nifti_resized}")
        resize_nifti_folder(source=dest_dir_nifti, dest=dest_dir_nifti_resized, image_shape=(resolution, resolution))

    # Normalize each volume
    dest_dir_nifti_resized_normalized = os.path.join(dest_dir, f'nifti_volumes_{resolution}x{resolution}_normalized')
    if process_nifti_normalized is not None:
        print(f"Normalize each volume, output folder: {dest_dir_nifti_resized_normalized}")
        normalize_folder(source=dest_dir_nifti_resized, dest=dest_dir_nifti_resized_normalized, dataset=dataset, modes_args=modes_args)

    # Write to pickle
    dest_dir_pkl = os.path.join(dest_dir, f'pickle_{resolution}x{resolution}_normalized')
    if snap_pickle is not None:
        print(f"Save to pickle, output_folder: {dest_dir_pkl}")
        print(f"-----")
        print("Each patient folder contains one .pkl file per slice")
        print("Each .pkl file contains all the modes associated to the current slice and the current patient")
        print(f"-----")
        convert_dataset_mi(source=dest_dir_nifti_resized_normalized, dest=dest_dir_pkl, max_patients=np.inf, dataset=dataset)

    # Save as zip
    dest_dir_zip = os.path.join(dest_dir, f'zip_{resolution}x{resolution}_normalized')
    if snap_zip is not None:
        print(f"Normalize each volume, output folder: {dest_dir_nifti_resized_normalized}")
        write_to_zip(source= os.path.join(dest_dir_pkl, 'temp'), dest=dest_dir_zip,  max_patients=100000, split=validation_args['split'])

#----------------------------------------------------------------------------

if __name__ == "__main__":

    # Configuration file
    print("Upload configuration file")
    with open('./configs/pelvic_preprocessing.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    id_exp = cfg['id_exp']
    source_dataset_name = cfg['data']['source_dataset']
    res = cfg['data']['image_size']
    modes_args = cfg['data']['modes']
    validation_args = cfg['data']['validation']

    # # Submit run:
    # print("Submit run")
    # # Get new id_exp
    # util_general.create_dir(os.path.join('log_run', source_dataset_name))
    # log_dir = os.path.join('log_run', source_dataset_name, cfg['network']['model_name'])
    # util_general.create_dir(log_dir)
    # # Save the configuration file
    # with open(os.path.join(log_dir, 'configuration.yaml'), 'w') as f:
    #     yaml.dump(cfg, f, default_flow_style=False)
    # # Initialize Logger
    # logger = util_general.Logger(file_name=os.path.join(log_dir, 'log.txt'), file_mode="w", should_flush=True)
    # # Copy the code in log_dir
    # files = util_general.list_dir_recursively_with_ignore('src', ignores=['.DS_Store', 'models'],
    #                                                       add_base_to_relative=True)
    # files = [(f[0], os.path.join(log_dir, f[1])) for f in files]
    # util_general.copy_files_and_create_dirs(files)

    # Welcome
    from datetime import datetime

    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print("Hello!", date_time)

    # Seed everything
    print("Seed all")
    util_general.seed_all(cfg['seed'])

    # Useful print
    print(f"Source dataset: {source_dataset_name}")
    print(f"Resolution {res}")
    print(f"Modes {list(modes_args.keys())}")
    print(f"Validation name: {validation_args['name']}")
    print(f"Validation splits: {validation_args['split']}")

    # Files and Directories
    print('Create file and directory')
    data_dir = os.path.join(cfg['data']['data_dir'], source_dataset_name)
    interim_dir = os.path.join(cfg['data']['interim_dir'], source_dataset_name)
    reports_dir = os.path.join(cfg['data']['reports_dir'], source_dataset_name)

    prepare_Pelvis_2_1(dataset = source_dataset_name, resolution=res, source_dir=data_dir, dest_dir=interim_dir, modes_args=modes_args, validation_args = validation_args, **cfg['data']['preprocessing_option'])
