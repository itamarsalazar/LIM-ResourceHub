import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, '/nfs/privileged/isalazar/projects/ultrasound-image-formation/')
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from src.US_data import LoadData_phantomLIM_ATSmodel539, LoadDataPICMUS
from exploration.Journal2023.strohm_train import load_model as load_strohm_model
from exploration.Journal2023.utils import downsample_channel_data
from src.beamforming_utils import make_pixel_grid_from_pos


def create_phantom_bmodes(h5name, depth_ini, model):
    h5_dataset_dir = f'{basic_dataset_dir}/verasonics'
    # depth_ini = get_data_from_name(h5name)
    P = LoadData_phantomLIM_ATSmodel539(h5_dir=h5_dataset_dir, h5_name=h5name)
    max_value = np.max(np.abs(np.array([P.idata, P.qdata])))
    P.idata = P.idata / max_value
    P.qdata = P.qdata / max_value

    laterals = np.linspace(P.phantom_xlims[0], P.phantom_xlims[-1], num=128)
    depths = np.linspace(depth_ini, depth_ini + 50, num=800) / 1000
    P.grid_xlims = P.phantom_xlims
    P.grid_zlims = np.array([depth_ini, depth_ini + 50]) / 1000
    # Downsample channel data
    channel_data_phantom = downsample_channel_data(copy.deepcopy(P),
                                                   laterals=laterals,
                                                   depths=depths,
                                                   device=device)
    channel_data_phantom = channel_data_phantom / channel_data_phantom.abs().max()
    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], 128)
    grid_full = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)

    bmode = model(channel_data_phantom)

    output_in_bmode_format = lambda x: np.clip((x * 60 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)

    bmode = output_in_bmode_format(bmode)

    return bmode, grid_full


def plot_phantom_bmodes(h5name, depth_ini, bmode_std, bmode_adv, rois, grid_full, metrics_std, metrics_adv):
    grid_xlims = grid_full[0, :, 0]
    grid_zlims = grid_full[:, 0, 2]
    extent_full = [grid_xlims[0] * 1e3, grid_xlims[-1] * 1e3,
                   grid_zlims[-1] * 1e3, grid_zlims[0] * 1e3]
    opts = {"extent": extent_full, "origin": "upper"}

    vmin = -50
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2 * (2), 4), sharey=True)
    # ax[0].imshow(bmode_std, cmap="gray", vmin=-60, vmax=0, **opts)
    ax[0].imshow(bmode_std, cmap="gray", vmin=vmin, vmax=0, **opts)
    ax[1].imshow(bmode_adv, cmap="gray", vmin=vmin, vmax=0, **opts)

    # roi_in, roi_ext = rois['in'], rois['ext']
    # ax[0].contour(roi_in, [0.5], colors="c", **opts)
    # ax[0].contour(roi_ext, [0.5], colors="m", **opts)
    # ax[1].set_title(
    #     f"STD \ngCNR: {metrics_std['gcnr']}, \ncnr: {metrics_std['cnr']}, \nsnr: {metrics_std['snr']}, \ncontrast: {metrics_std['contrast']}")
    # ax[2].set_title(
    #     f"ADV \ngCNR: {metrics_adv['gcnr']}, \ncnr: {metrics_adv['cnr']}, \nsnr: {metrics_adv['snr']}, \ncontrast: {metrics_adv['contrast']}")

    ax[0].set_title("Standard\nTraining")
    ax[1].set_title("Adversarial\nTraining")
    ax[0].set_xlabel('Lateral (mm)')
    ax[0].set_ylabel('Axial (mm)')
    ax[1].set_xlabel('Lateral (mm)')

    plt.suptitle(f"{h5name}, depth_ini: {depth_ini} mm")
    plt.tight_layout()

    plt.suptitle(f"{h5name}")
    plt.tight_layout()
    plt.savefig(f"{this_dir}/{h5name}_A.png")
    plt.close()


from src.metrics_utils import contrast, cnr, gcnr, snr


def compute_metrics(bmode, roi_in, roi_ext):
    if roi_in is None:
        metrics = {
            'contrast': None,
            'snr': None,
            'gcnr': None,
            'cnr': None
        }
        return metrics

    # Compute metrics
    env = 10 ** (bmode / 20)
    env_in = env[roi_in]
    env_ext = env[roi_ext]
    contrast_value = np.abs(contrast(env_in, env_ext))
    snr_value = snr(env_ext)
    gcnr_value = gcnr(env_in, env_ext)
    cnr_value = np.abs(cnr(env_in, env_ext))
    # psnr_value = psnr(bmode, bmodeRef)

    metrics = {
        'contrast': contrast_value,
        'snr': snr_value,
        'gcnr': gcnr_value,
        'cnr': cnr_value
    }

    return metrics

def get_rois(xctr, zctr, r, grid_full):
    if xctr is None:
        return None, None
    dist = np.sqrt((grid_full[:, :, 0] - xctr) ** 2 + (grid_full[:, :, 2] - zctr) ** 2)
    r0, r1 = r - 0.5/1000, r + 0.5/1000,
    r2 = np.sqrt(r0**2 + r1**2)
    # roi_cyst = dist <=r
    roi_in = dist <= r0
    roi_ext = (r1 <= dist) * (dist <= r2)
    # roi_ext = r1 <= dist
    return roi_in, roi_ext


def get_rois_dict(h5name, grid_full):
    if h5name == 'IS_L11-4v_data2_RF.h5':
        xctrA, zctrA = -9 / 1000, 50.25 / 1000
    elif h5name == 'IS_L11-4v_data3_RF.h5':
        xctrA, zctrA = 17 / 1000, 50.25 / 1000
    elif h5name == 'IS_L11-4v_data1_RF.h5':
        xctrA, zctrA = 20 / 1000, 50.25 / 1000
    else:
        raise NotImplementedError

    r = 7.5 / 1000
    roi_in_A, roi_ext_A = get_rois(xctrA, zctrA, r, grid_full)

    # roi_ext_A = roi_ext_A*(~roi_in_B)

    rois = {'in': roi_in_A, 'ext': roi_ext_A}
    return rois



device = 'cuda' if torch.cuda.is_available() else 'cpu'
this_dir = 'C:/Users/u_imagenes/PycharmProjects/ultrasound-image-formation/exploration/IUS2024/'
basic_dataset_dir = 'F:/Itamar_LIM/datasets/'
epoch = 99
das_label = 'DAS'

model_strohm_std, _ = load_strohm_model(model_dir=os.path.join(this_dir, "models", 'strohm_cyclic'),
                                    epoch=89,
                                    device=device)
model_strohm_adv, _ = load_strohm_model(model_dir=os.path.join(this_dir, "models", 'strohm_adv'),
                                        epoch=epoch,
                                        device=device)
model_strohm_std.eval()
model_strohm_adv.eval()
print("")

h5name_list = ['IS_L11-4v_data2_RF.h5', 'IS_L11-4v_data3_RF.h5', 'IS_L11-4v_data1_RF.h5']
# h5name = 'IS_L11-4v_data3_RF.h5'
for h5name in h5name_list:
    # depth_ini = 30
    depth_ini = 30  # 'IS_L11-4v_data3_RF.h5'
    phantom_std, grid_full = create_phantom_bmodes(h5name, depth_ini, model_strohm_std)
    phantom_adv, _ = create_phantom_bmodes(h5name, depth_ini, model_strohm_adv)

    rois = get_rois_dict(h5name, grid_full)
    print("Metrics for inclusion:")
    roi_in, roi_ext = rois['in'], rois['ext']
    metrics_std = compute_metrics(phantom_std, roi_in=roi_in, roi_ext=roi_ext)
    metrics_adv = compute_metrics(phantom_adv, roi_in=roi_in, roi_ext=roi_ext)
    print(f"STD: {metrics_std}")
    print(f"ADV: {metrics_adv}")

    # plot_phantom_bmodes(h5name, depth_ini, phantom_std, phantom_adv,
    #                     rois,
    #                     grid_full)
    plot_phantom_bmodes(h5name, depth_ini, phantom_std, phantom_adv, rois, grid_full, metrics_std, metrics_adv)