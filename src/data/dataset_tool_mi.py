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
from src.utils import util_general

# Minh
import glob
import dicom2nifti
from itertools import repeat
import nibabel as nib
from multiprocessing import Pool
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
            image = utils.read_image(file_mode, image_shape=image_shape, interpolation=interpolation, crop=None) # read and reshape image # todo step are not clear inside this fun
            nib.save(image, output_file)
        except:
            raise IOError(f"fail to convert {file_mode:s}")

def resize_nifti_folder(source: str, dest: str, image_shape=(256, 256)):  # rescale from [512, 512, n_slices_patient] -> [res, res, n_slices_patient]


    folders = glob.glob(os.path.join(source, "*"))

    # try:
    #     pool = Pool()  # Multithreading
    #     l = pool.starmap(resize_file, zip(range(len(folders)), repeat(folders), repeat(dest), repeat(image_shape))) # todo check
    #     pool.close()
    # except:
    #     for idx_pat in range(len(folders)):
    #         resize_file(folder_index=idx_pat, folders=folders, dest=dest, image_shape=image_shape)
    for idx_pat in range(len(folders)):
        resize_file(folder_index=idx_pat, folders=folders, dest=dest, image_shape=image_shape)

#----------------------------------------------------------------------------

def get_dataset_modality(file, dataset):
    if dataset == "Pelvis_2.1":
        if (
            "MR_MR_T2.nii.gz" in file
            or "MR_MR_T2_BC.nii.gz" in file
            or "MR_MR_T2_Fat.nii.gz" in file
            or "MR_MR_T2_OutPhase.nii.gz" in file
            or "MR_MR_T2_Water.nii.gz" in file
        ):
            return "MR"
        elif "MR_Bias_Field.nii.gz" in file:
            return "bias"
        else:
            return "CT"
    elif dataset == "brats20":
        if "truth.nii.gz" in file:
            return "truth"
        else:
            return "MR"

def normalize_file(folder_index, folders, dest, dataset, low=0.0, hi=255.0):
    def normalize_per_dataset(data, dataset):
        if dataset == "brats20":
            # upper, lower = volume_utils.get_percentile(data) # Can not differentiate between different tumor regions.
            upper, lower = data.max(), data.min()
            data = np.clip(data, lower, upper)
            data = (data - lower) / (upper - lower)
            data = data * (hi - low) + low
        elif dataset == "Pelvis_2.1":
            """TODO
            4000 for upper?
            """
            upper, lower = data.max(), 0.0  # use 4000 instead of data.max() for upper?
            data = np.clip(data, lower, upper)
            data = (data - lower) / (upper - lower)
            data = data * (hi - low) + low
        else:
            raise NotImplementedError(
                f"Normalization for {dataset} was not implemented."
            )
        return data

    folder = folders[folder_index]
    name = path_utils.get_filename_without_extension(folder)
    files = glob.glob(os.path.join(folder, "*"))

    # Get all modalitites.
    if os.path.isdir(files[0]):
        files = glob.glob(os.path.join(files[0], "*"))

    # For each modality.
    for file in files:
        output_dir = os.path.join(dest, name)
        path_utils.make_dir(output_dir)  # Make directory for each patient if not exist.

        fname = path_utils.get_filename_without_extension(file)
        output_file = os.path.join(output_dir, f"{fname}.gz")

        try:
            print(f"Reading: {file}")
            volume = nib.load(file)
            data = volume.get_fdata()
            modality = get_dataset_modality(file, dataset)

            """follow this https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
            """
            if modality == "CT":
                percentile_99_5, percentile_00_5 = volume_utils.get_percentile(data)
                lower_bound, upper_bound = -1000, 2000
                data = np.clip(data, lower_bound, upper_bound)
                # Map to [-1, 1].
                # data = (data - 500) / 1500 # if you want to map to [-1,1]
                data = (data + 1000) / 3000  # if you want to map to [0,1]
            elif modality in ["truth", "seg"]:  # groundtruth
                pass
            else:  # MRI
                data = normalize_per_dataset(data, dataset)

            affine = volume.affine
            image = nib.Nifti1Image(data, affine=affine)
            nib.save(image, output_file)

        except:
            raise IOError(f"fail to normalize {file:s}")

def normalize_folder(source: str, dest: str, dataset: str):
    folders = glob.glob(os.path.join(source, "*"))

    try:
        pool = Pool() # Multithreading
        l = pool.starmap(
            normalize_file,
            zip(range(len(folders)), repeat(folders), repeat(dest), repeat(dataset)),
        )
        pool.close()
    except:
        for idx_pat in range(len(folders)):
            normalize_file(folder_index=idx_pat, folders=folders, dest=dest, dataset=dataset)

#----------------------------------------------------------------------------

def prepare_Pelvis_2_1(source_dir, dest_dir,
                       modes_arg,
                       resolution,
                       process_dicom_2_nifti=None,
                       process_nifti_resized=None,
                       process_nifti_normalized=None,
                       snap_pickle=None,
                       snap_zip=None):

    # From dicom to nifti
    dest_dir_nifti = os.path.join(dest_dir, 'nifti_volumes')
    if process_dicom_2_nifti is not None:
        print(f"Convert to nifti, output folder: {dest_dir_nifti}")
        convert_dicom_2_nifti(source=source_dir, dest=dest_dir_nifti, modes_to_preprocess=list(modes_arg.keys()))

    # Resize nifti volume from [original_res, original_res, n_slices] to [res x res x n_slices]
    dest_dir_nifti_resized = os.path.join(dest_dir, f'nifti_volumes_{resolution}x{resolution}')
    if process_nifti_resized is not None:
        print(f"Resize to resolution {resolution}, output folder: {dest_dir_nifti_resized}")
        resize_nifti_folder(source=dest_dir_nifti, dest=dest_dir_nifti_resized, image_shape=(resolution, resolution))

    # Normalize each volume
    dest_dir_nifti_resized_normalized = os.path.join(dest_dir, f'nifti_volumes_{resolution}x{resolution}_normalized')  # todo the statistics to normalize are passed from config file
    if process_nifti_normalized is not None:
        print(f"Normalize each volume, output folder: {dest_dir_nifti_resized_normalized}")
        normalize_folder(source=dest_dir_nifti_resized, dest=dest_dir_nifti_resized_normalized, modes_arg=modes_arg, dataset="Pelvis_2.1") #todo

    # Write to pickle
    if snap_pickle is not None:
        pass #todo
    # Save as zip
    if snap_zip is not None:
        pass #todo

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
    print(f"Modes {list(modes_args.keys())}")
    print(f"Resolution {res}")

    # Files and Directories
    print('Create file and directory')
    data_dir = os.path.join(cfg['data']['data_dir'], source_dataset_name)
    interim_dir = os.path.join(cfg['data']['interim_dir'], source_dataset_name)
    reports_dir = os.path.join(cfg['data']['reports_dir'], source_dataset_name)

    # todo compute patient list and generate folder splits and set different experiment with an incremental number of patients

    prepare_Pelvis_2_1(source_dir=data_dir, dest_dir=interim_dir, modes_args=modes_args, resolution=res, **cfg['data']['preprocessing_option'])
