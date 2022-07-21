import dnnlib
import os
import numpy as np
from PIL import Image
import pickle
import torch
import time

from src.utils import util_general

def projection_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    projector_kwargs        = {},       # Optiond for Projector algorithm
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 1,        # Total batch size for one training iteration.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
):

    # Initialize.
    start_time = time.time()
    device = torch.device('cuda:1') #torch.device('cuda', rank) # device = torch.device('cuda:1')
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    # Load training set.
    print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Datasetx
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size//num_gpus, shuffle=False)
    print()
    print('Num images: ',   len(training_set))
    print('Image shape:',   training_set.image_shape)
    print('Label shape:',   training_set.label_shape)
    print()

    # Constructing Projector.
    if rank == 0:
        print('Constructing Projector...')
    projector = dnnlib.util.construct_class_by_name(
        class_name='genlib.projector.projector.StyleGAN2Projector', outdir=run_dir, device=device, modalities=training_set_kwargs.modalities, dtype=training_set_kwargs.dtype, **projector_kwargs
    )  # subclass of torch.nn.Module

    # todo add early stopping after n_iterations=500
    # todo add slice step selection
    # todo add option to provide an external starting latent point (from previuous inversion on same patient) z_<idp>_<ids>
    # todo Implement multiprocessing through GPU
    # todo overall loss plot for each patient->point on x axis is slices, point on y axis is loss
    projected_w = util_general.nested_dict()
    idx = 0
    temp = ''
    w_init = None
    for phase_real_img, _, fname in training_set_iterator:
        # Skip slice schedule.
        if idx % projector_kwargs.step_patient_slice == 0:

            # Send target image ([1, n_modalities, 256, 256]) to device.
            phase_real_img = phase_real_img.to(device).to(torch.float32)

            # Select info regarding patient and slice.
            _, id_patient, id_slice = util_general.split_dos_path_into_components(path=fname[0])
            id_slice = util_general.get_filename_without_extension(path=id_slice)[-5:]
            print(f'Patient:        {id_patient}')
            print(f'Slice:          {id_slice}')
            print('')

            # Smooth restarting schedule.
            if id_patient == temp:
                assert w_init is not None
                projected_w_tensor, best_step = projector.forward(
                    id_patient=id_patient, id_slice=int(id_slice), target=phase_real_img, num_steps=projector_kwargs.num_steps, early_stopping=projector_kwargs.early_stopping, w_init=w_init, verbose=projector_kwargs.verbose
                )
            else: # new patient selected
                projected_w_tensor, best_step = projector.forward(
                    id_patient=id_patient, id_slice=int(id_slice), target=phase_real_img, num_steps=1000, early_stopping=100000, verbose=False # num_steps=1000, early_stopping=100000, verbose=True
                )
                temp = id_patient
                w_init =  projected_w_tensor[best_step].detach().cpu()

            projected_w[id_patient][id_slice] = projected_w_tensor[best_step].detach().cpu().numpy() # projected_w_tensor[-1].detach().cpu().numpy()
        else:
            pass

        idx += 1

    with open(os.path.join(run_dir,'projected_w'), 'wb') as handle:
        pickle.dump(projected_w, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------------------------------------------------------------------

def projection_test(
        target_fname =           '.',    # Path to image to invert
        run_dir =                '.',    # Output directory.
        training_set_kwargs =    {},     # Options for training set.
        projector_kwargs =       {},     # Optiond for Projector algorithm
        random_seed =            0,      # Global random seed.
        num_gpus =               1,      # Number of GPUs participating in the training.
        rank =                   0,      # Rank of the current process in [0, num_gpus[.
        cudnn_benchmark =        True,   # Enable torch.backends.cudnn.benchmark?
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)  # torch.device('cuda', rank) # device = torch.device('cuda:1')
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.

    # Load single Target Image.
    print('Loading Test Image...')

    target_pil = Image.open(target_fname)
    if target_pil.format != 'TIFF':
        target_pil = Image.open(target_fname).convert('L')  # convert in grayscale
    target_numpy = np.array(target_pil)
    if target_numpy.max() <= 1:
        print(f"Unsupported max value : Expected values in [0, 255] but got [{target_numpy.min()}, {target_numpy.max()}]!")
        target_numpy = target_numpy * 255.0
    target = torch.Tensor(target_numpy)  # convert to Torch Tensor
    if target.ndim != 4:
        print(f"Unsupported input dimension: Expected 4D (batched) input but got input of size: {target.shape}! Let's add dimension!")
        for _ in range(4 - target.ndim):
            target = torch.unsqueeze(target, dim=0)
        target = target.view([1, 1, 256, 256])
    if target.dtype != torch.float32:
        print( f"Unsupported input type: Expected float32 but got {target.dtype}! Let's change Tensor dtype!")
        target = target.to(torch.float32)
    print()
    print('Image shape:', target.shape)
    print('Image dtype:', target.dtype)
    print('Image min:', target.min().item())
    print('Image max:', target.max().item())
    print()

    # Constructing Projector.
    if rank == 0:
        print('Constructing Projector...')
    projector = dnnlib.util.construct_class_by_name(
        class_name='genlib.projector.projector.StyleGAN2Projector', outdir=run_dir, device=device, modalities=training_set_kwargs['modalities'], dtype=training_set_kwargs['dtype'], **projector_kwargs
    )  # subclass of torch.nn.Module

    idp, ids = util_general.get_filename_without_extension(target_fname).split('_')
    _ = projector.forward(id_patient = idp, id_slice = int(ids), target = target, verbose = True)
    print("May be the force with you!")

if __name__ == "__main__":
    # For debug.
    my_env = os.environ.copy()
    my_env["PATH"] = "/home/lorenzo/miniconda3/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)

    # Dataset info.
    training_set_kwargs = {
        'modalities': ['CT'],
        'dtype': 'float32'
    }
    # Target image.
    dataset = 'claro_retrospettivo_no_casting'
    interim_dir = '/home/lorenzo/Gan-track/data/interim/'
    idp = '100151470'
    ids = '103'
    report_dir = '/home/lorenzo/Gan-track/reports/'
    experiment = '00000'

    # Network info.
    network_pkl = 'network-snapshot-004400.pkl'

    outdir_model = os.path.join(report_dir, dataset, 'training_runs', dataset, training_set_kwargs['modalities'][0], f'{experiment}-stylegan2-stylegan2-ada-gpus2-batch32-gamma0.4096', network_pkl)
    target_fname =  os.path.join(interim_dir, dataset, 'stylegan2-ada', f'{idp}_{ids}.tif')
    w_pix_list = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    w_lips = 1.0
    for idx_exp, w_pix in enumerate(w_pix_list):
        run_dir = os.path.join(report_dir, dataset, 'projection-runs', dataset,  training_set_kwargs['modalities'][0], f'{idx_exp:05d}-stylegan2-w_lips_{w_lips}-w_pix_{w_pix}-{network_pkl}')

        projector_kwargs = {
            'outdir_model': outdir_model,
            'num_steps': 1000,
            'early_stopping': 1000,
            'w_lpips': w_lips,
            'w_pix': w_pix,
            'save_final_projection': True,
            'snap_video': 1,
            'snap_optimization_history': 1
        }

        projection_test(
            target_fname = target_fname,
            run_dir = run_dir,
            training_set_kwargs = training_set_kwargs,
            projector_kwargs = projector_kwargs
        ) # pylint: disable=no-value-for-parameter