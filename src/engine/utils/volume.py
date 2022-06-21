import os
import ntpath
import numpy as np
import nibabel as nib
import itertools
import scipy

from nilearn.masking import compute_multi_background_mask
from src.engine.utils.path_utils import get_modality


def get_spacing(volume):
    return volume.header.get_zooms()


def get_shape(volume):
    return volume.shape


def get_bounding_box_nd(volume):
    N = volume.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(volume, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def get_percentile(volume, p=99.5):
    percentile_p = np.percentile(volume, p)
    percentile_1_p = np.percentile(volume, 100 - p)
    return percentile_p, percentile_1_p


def get_bounding_box(volume):
    r = np.any(volume, axis=(1, 2))
    c = np.any(volume, axis=(0, 2))
    z = np.any(volume, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return np.array([rmin.T, rmax.T, cmin.T, cmax.T, zmin.T, zmax.T])


def get_size_bounding_box(volume):
    rmin, rmax, cmin, cmax, zmin, zmax = get_bounding_box(volume)
    return np.array([rmax - rmin, cmax - cmin, zmax - zmin])


def get_non_zeros_pixel(volume):
    return np.count_nonzero(volume)


def get_zeros_pixel(volume):
    return volume.size - np.count_nonzero(volume)


def compute_mean_non_zeros_pixel(volume):
    return volume[volume != 0].mean()


def compute_std_non_zeros_pixel(volume):
    return np.nanstd(np.where(np.isclose(volume, 0), np.nan, volume))


def count_number_occurrences_label(truth):
    temp_truth = truth.astype(int)
    truth_reshape = temp_truth.ravel()
    return np.bincount(truth_reshape)


def get_unique_label(truth):
    temp_truth = truth.astype(int)
    truth_reshape = temp_truth.ravel()
    return np.unique(truth_reshape)


def get_max_min_intensity(volume):
    return np.max(volume), np.min(volume), np.min(volume[volume != 0])


def get_size(volume_path):
    return round(os.path.getsize(volume_path) / 1000000, 1)


def count_non_zeros_background(volume, truth):
    indice_volume = volume > 0
    indice_truth = truth == 0
    indice = np.multiply(indice_volume, indice_truth)
    return np.count_nonzero(indice)


def count_zeros_non_background(volume, truth):
    indice_volume = volume <= 0
    indice_truth = truth != 0
    indice = np.multiply(indice_volume, indice_truth)
    return np.count_nonzero(indice)


def get_filename_with_extenstion(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filename_without_extenstion(path):
    filename = get_filename_with_extenstion(path)
    return filename.replace(".nii.gz", "")


def get_truth_path(volume_path, truth_name="truth"):
    volume_filename = get_filename_without_extenstion(volume_path)
    truth_path = volume_path.replace(volume_filename, truth_name)
    return truth_path


def get_volume_paths(config, truth_path, truth="truth_name"):
    volume_paths = list()
    for modality in config["training_modalities"]:
        volume_path = truth_path.replace(truth, modality)
        volume_paths.append(volume_path)
    return volume_paths


def get_volume_paths_from_one_volume(volume_path, training=["T1"]):
    volume_paths = list()
    volume_modality = get_modality(volume_path)
    for modality in training:
        temp_path = volume_path.replace(volume_modality, modality)
        volume_paths.append(temp_path)
    return volume_paths


def is_truth_path(path, truth_name="truth"):
    if truth_name in path:
        return True
    else:
        return False


def get_background_mask(volume_paths):
    """
    This function computes a common background mask for all of the data in a subject folder.
    :param input_dir: a subject folder from the BRATS dataset.
    :param out_file: an image containing a mask that is 1 where the image data for that subject contains the background.
    :param truth_name: how the truth file is labeled int he subject folder
    :return: the path to the out_file
    """
    volumes_data = list()
    for path in volume_paths[0]:
        volume = nib.load(path)
        volumes_data.append(volume)

    background_image = compute_multi_background_mask(volumes_data)
    return background_image


def get_depth_one_region(truth_original, label):
    truth = truth_original.copy()
    truth[truth != label] = 0
    truth[truth == label] = 1
    _, _, _, _, zmin, zmax = get_bounding_box(truth)
    return zmax - zmin


def get_depth_one_label(truth_original, label):
    truth = truth_original.copy()
    truth[truth != label] = 0
    truth[truth == label] = 1

    labeled_array, num_features = scipy.ndimage.label(truth)

    depth_list = list()
    depth_mean = 0
    for region_i in range(1, num_features + 1, 1):
        depth = get_depth_one_region(labeled_array, region_i)
        depth_mean += depth
        depth_list.append(depth)

    # return [min(depth_list), sum(depth_list)/len(depth_list), max(depth_list)]
    try:
        return max(depth_list)
    except:
        return 0


def compute_displament(center_curr, center_prev):
    return np.sqrt(
        (center_curr[0] - center_prev[0]) ** 2 + (center_curr[1] - center_prev[1]) ** 2
    )


def compute_average_center_of_mass(center_curr):
    return tuple([sum(y) / len(y) for y in zip(*center_curr)])


def get_spatial_displament_one_label(truth_original, label):
    truth = truth_original.copy()
    truth[truth != label] = 0
    truth[truth == label] = 1

    _, _, depth = truth.shape

    slice_type = "zero"
    displament = list()
    for d in range(depth):
        d_slice = truth[:, :, d]
        if get_non_zeros_pixel(d_slice) > 0:
            center_curr = scipy.ndimage.measurements.center_of_mass(d_slice)

            if isinstance(center_curr, list):
                center_curr = compute_average_center_of_mass(center_curr)

            if slice_type == "zero":
                center_prev = center_curr
                slice_type = "object"
            else:
                displament.append(compute_displament(center_curr, center_prev))
                center_prev = center_curr
    return displament


def get_percentage_label_total_voxels(truth_original, label):
    truth = truth_original.copy()
    truth[truth != label] = 0
    truth[truth == label] = 1
    return get_non_zeros_pixel(truth) / truth.size


def get_info_all_organs(truth, challenge="ibsr"):
    info = list()
    if challenge == "brats19":
        n_labels = 4
    else:
        n_labels = len(get_unique_label(truth))

    for label in range(1, n_labels, 1):
        s = get_spatial_displament_one_label(truth, label)
        s = sum(s) / len(s)
        info_label = [
            get_depth_one_label(truth, label),
            get_percentage_label_total_voxels(truth, label),
            s,
        ]
        info.extend(info_label)
    return info


def get_spatial_displament_all_organs(truth, challenge="ibsr"):
    info = list()
    if challenge == "brats19":
        n_labels = 4
    else:
        n_labels = len(get_unique_label(truth))

    for label in range(1, n_labels, 1):
        info_label = get_spatial_displament_one_label(truth, label)
        try:
            info.append(sum(info_label) / len(info_label))
        except:
            info.append(0)

    return info
