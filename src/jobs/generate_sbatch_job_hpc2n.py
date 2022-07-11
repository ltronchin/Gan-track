import dnnlib
import itertools


def replace(filedata, string, bystring):
    filedata = filedata.replace(string, bystring)
    return filedata

"""
def prepare_dataset(
    filedata, dataset, num_patients, model, aug, mirror, in_modal, out_modal
):
"""
def prepare_dataset(
       filedata,
       template_args,
):

    assert len(template_args) >= 1, "modalities is empty"

    filename = f"./src/bash/{template_args.dataset:s}-num_{template_args.num_patients:d}-modalities_{template_args.modalities:s}-aug_{template_args.aug:s}-aug_opts_{template_args.aug_opts:s}.sh"

    for key in list(template_args.keys()):
        arg = str(template_args[key])
        print(f"Replace <{key}> with {arg}")
        filedata = replace(filedata, f"<{str(key)}>", arg)

    # Write the file out again
    with open(filename, "w") as file:
        file.write(filedata)
    print(f"sbatch {filename}")


def main():
    # Read in the file
    with open(f"./configs/bash/template_stylegan3.sh", "r") as file:
        template = file.read()

    # Parameter to fix for template
    """
        --outdir=<outdir>
        --data=<source_path>/<dataset>/<dataset>-num-<num_patients>_train-0.70_val-0.20_test-0.10.zip
        --dataset=<dataset>
        --split=<split>
        --modalities=<modalities>
        --dtype=<dtype>
        --cfg=<model>
        --batch=<batch>
        --map-depth=<map_depth>
        --glr=<glr>
        --dlr=<dlr>
        --cbase=<cbase>
        --kimg=<kimg>
        --gpus=<n_gpus>
        --workers=<workers>
        --gamma=<gamma>
        --snap=<snap>
        --mirror=<mirror>
        --aug=<aug>
        --ada_kimg=<ada_kimg>
        --aug_opts=<aug_opts>
        --target=<target_p_aug>
        --metrics=<metrics>
        --metrics_cache=<metrics_cache>"
    """

    # Transformations options. (rotate90 exluded)
    pixel_blitting = 'xflip,xint' # 'xflip,rotate90,xint'
    general_geometric = 'scale,rotate,aniso,xfrac' # rotate range limited to [-2 2] degree
    color = '' # 'brightness,contrast'
    image_space_filtering = ''
    image_space_corruptions = ''

    run_dict = {
        'runset_1': {
            'aug_opts': [
                'noaug',  # no aug case
                pixel_blitting,
                pixel_blitting + ',' + general_geometric,
                #pixel_blitting + ',' + color,
                general_geometric,
                #general_geometric + ',' + color,
                #general_geometric + ',' + color,
                #pixel_blitting + ',' + general_geometric + ',' + color
            ],
            'modalities': [
                'MR_nonrigid_CT,MR_MR_T2'
            ]
        },
        'runset_2': {
            'aug_opts': [
                'noaug',
            ],
            'modalities': [
                'MR_nonrigid_CT',
                'MR_MR_T2'
            ]
        }
    }

    c = dnnlib.EasyDict()  # Main config dict.
    for run_key in list(run_dict.keys()):
        for opts in itertools.product(*run_dict[run_key].values()):
            opt_aug = opts[0]
            opt_modalities = opts[1]
            print(f'Opt aug: {opt_aug}, opt modalities: {opt_modalities}')

            # Dataset and data folder options.
            c.outdir                = '/pfs/proj/nobackup/fs/projnb10/snic2020-6-234/lotr/Gan-track/reports'
            c.source_path           = '/pfs/proj/nobackup/fs/projnb10/snic2020-6-234/lotr/Gan-track/data/interim'
            c.dataset               = 'Pelvis_2.1'
            c.split                 = 'train'
            c.num_patients          =  375
            c.modalities            =  opt_modalities #'MR_nonrigid_CT,MR_MR_T2'
            c.dtype                 = 'float32'

            # Model options.
            c.model                 = 'stylegan2'
            c.batch                 = 16
            c.map_depth             = 2
            c.glr                   = 0.0025
            c.dlr                   = 0.0025
            c.cbase                 = 16384

            # Training options.
            c.kimg                  = 10000
            c.gpus                  = 1
            c.workers               = 3
            c.gamma                 = 0.8192
            c.snap                  = 10
            c.mirror                = 1

            # Augmentation options.
            if opt_aug == 'noaug':
                c.aug                   = 'noaug'
                c.ada_kimg              = 500
                c.aug_opts              = 'noaug'
                c.rotate_max            = 0
                c.target                = 0
            else:
                c.aug                   = 'ada'
                c.ada_kimg              = 500
                c.aug_opts              = opt_aug
                c.rotate_max            = 5
                c.target                = 0.6

            # Metrics options.
            c.metrics                   = 'fid50k_full'
            c.metrics_cache             = True

            prepare_dataset(filedata=template, template_args=c)

    print("\n")
    print("May be the force with you.")

if __name__ == "__main__":
    main()
