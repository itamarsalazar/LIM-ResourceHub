import os
import torch
import numpy as np
import torch.nn as nn
from src.beamforming_utils import make_pixel_grid_from_pos
from scipy.signal import convolve2d
from src.beamforming_Goudarzi import goudarzi_MobileNetV2, goudarzi_MobileNetV2_w16
import random
from scipy.ndimage import binary_erosion, gaussian_filter, binary_dilation
from src.beamforming_Rothlubbers import rothlubbersModelSingleWeight_rawdomain
from src.metrics_utils import contrast, cnr, gcnr, snr, psnr
from src.models.unet.unet_model import UNet_nair2020
from scipy.interpolate import RectBivariateSpline
# from src.models.dataloaders.cystDataset import create_input_Id
from scipy import interpolate
from pytorch_msssim import MS_SSIM
from skimage.restoration import denoise_nl_means


def load_roth_model(project_dir, H, grid, dgs_double, device):
    model_dir = os.path.join(project_dir, "cubdl", "submissions", "rothlubbers")
    weights_file = os.path.join(model_dir, "task1_bfFinal_CSW2D_stateDict.pth")
    model = rothlubbersModelSingleWeight_rawdomain(H, grid, DGS_DOUBLE=dgs_double, device=device).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))
    model.eval()
    return model


def load_goud_model(H, grid, ipixels, project_dir, device, dgs_double=False):
    model_dir = os.path.join(project_dir, "cubdl", "submissions", "goudarzi")
    model_name = "model_weights.pt"
    cval = 1e-5
    model = goudarzi_MobileNetV2(H, grid, ipixels=ipixels, device=device, cval=cval, DGS_DOUBLE=dgs_double).to(device)
    # model = goudarzi_MobileNetV2_w16(P, grid, ipixels=ipixels, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location=torch.device(device)))
    model.eval()

    return model


def load_nair_model(model_dir, model_name, device):
    model = UNet_nair2020(n_channels=2, n_classes=1).to(device)
    model.load_state_dict(torch.load('%s/%s' % (model_dir, model_name), map_location=device))
    model.eval()
    return model


def make_bimg_nair_roi(H, model, grid, device):
    # H.idata = H.idata / np.amax(H.idata)
    # H.qdata = H.qdata / np.amax(H.qdata)

    H.idata = H.idata / np.abs(H.idata).max()
    H.qdata = H.qdata / np.abs(H.qdata).max()

    # x = np.stack((H.idata.squeeze().T, H.qdata.squeeze().T), axis=0)
    # # x.shape = 2, 800, 128
    # x = torch.from_numpy(x)[None, :].to(torch.float).to(device)
    # # x.shape = 1, 2, 800, 128

    depths = np.linspace(30, 80, num=800) / 1000
    # depths = np.linspace(grid[0, 0, 2], grid[-1, 0, 2], num=800) / 1000
    # depths = grid[:, 0, 2]
    print(len(depths))
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)

    # print('<NAIR')
    bimg, seg = model(x)
    # print('\>NAIR')
    bimg = bimg*60-60
    bimg = bimg.detach().cpu().numpy().squeeze()
    seg = seg.detach().cpu().numpy().squeeze()

    bbox_orig = (H.grid_xlims[0], H.grid_xlims[-1], H.grid_zlims[0], H.grid_zlims[-1])
    bbox_new = (grid[0, 0, 0], grid[0, -1, 0], grid[0, 0, 2], grid[-1, 0, 2])

    size_output = (grid.shape[1], grid.shape[0])

    bimg_roi = interpolate_data(bimg, bbox_orig=bbox_orig, bbox_new=bbox_new, size_output=size_output)
    seg_roi = interpolate_data(seg, bbox_orig=bbox_orig, bbox_new=bbox_new, size_output=size_output)

    # rows, cols = grid.shape[0], grid.shape[1]
    # # grid: [nrows, ncols, 3]
    # # grid: [200, 32, 3]
    #
    # bimg = np.resize(bimg, (rows, cols))
    # seg = np.resize(seg, (rows, cols))

    return bimg_roi, seg_roi, bimg, seg


def make_bimg_unet1deco(H, model, grid, device):
    H.idata = H.idata / np.abs(H.idata).max()
    H.qdata = H.qdata / np.abs(H.qdata).max()

    depths = np.linspace(30, 80, num=800) / 1000
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)

    bimg = model(x)
    bimg = bimg*60-60
    bimg = bimg.detach().cpu().numpy().squeeze()

    bbox_orig = (H.grid_xlims[0], H.grid_xlims[-1], H.grid_zlims[0], H.grid_zlims[-1])
    bbox_new = (grid[0, 0, 0], grid[0, -1, 0], grid[0, 0, 2], grid[-1, 0, 2])

    size_output = (grid.shape[1], grid.shape[0])

    bimg_roi = interpolate_data(bimg, bbox_orig=bbox_orig, bbox_new=bbox_new, size_output=size_output)

    return bimg_roi, bimg


def make_bimg_strohm(H, model, device):
    H.idata = H.idata / np.abs(H.idata).max()
    H.qdata = H.qdata / np.abs(H.qdata).max()

    depths = np.linspace(30, 80, num=1600) / 1000
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)

    bimg = model(x)
    bimg = bimg*60-60
    bimg = bimg.detach().cpu().numpy().squeeze()

    return bimg

def make_bimg_strohm_roi(H, model, grid, device):
    H.idata = H.idata / np.abs(H.idata).max()
    H.qdata = H.qdata / np.abs(H.qdata).max()

    depths = np.linspace(30, 80, num=1600) / 1000
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)

    bimg = model(x)
    bimg = bimg*60-60
    bimg = bimg.detach().cpu().numpy().squeeze()

    bbox_orig = (H.grid_xlims[0], H.grid_xlims[-1], H.grid_zlims[0], H.grid_zlims[-1])
    bbox_new = (grid[0, 0, 0], grid[0, -1, 0], grid[0, 0, 2], grid[-1, 0, 2])

    size_output = (grid.shape[1], grid.shape[0])

    bimg_roi = interpolate_data(bimg, bbox_orig=bbox_orig, bbox_new=bbox_new, size_output=size_output)

    return bimg_roi, bimg


def make_bimg_wang(H, model, device):
    H.idata = H.idata / np.abs(H.idata).max()
    H.qdata = H.qdata / np.abs(H.qdata).max()

    depths = np.linspace(30, 80, num=800) / 1000
    channel_data, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    channel_data = torch.from_numpy(channel_data)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)

    N, C, H, W = channel_data.size()
    z = torch.randn(N, 1, H, W).to(device)

    # print(f"channel_data: {channel_data.size()}")
    # print(f"(Size) z: {z.size()}")
    # print(f"(Size) cat: {torch.cat((channel_data, z), dim=1).size()}")

    bimg = model(torch.cat((channel_data, z), dim=1))
    bimg = bimg*60-60
    bimg = bimg.detach().cpu().numpy().squeeze()

    return bimg


def make_bimg_nair(H, model, device):
    # I renamed this functions so...
    # Perhaps you are looking for "make_bimg_nair_roi"

    # H.idata = H.idata / np.abs(H.idata).max()
    # H.qdata = H.qdata / np.abs(H.qdata).max()
    norm_value = np.maximum(np.abs(H.idata).max(), np.abs(H.qdata).max())
    H.idata = H.idata / norm_value
    H.qdata = H.qdata / norm_value

    depths = np.linspace(30, 80, num=800) / 1000
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)

    bimg = model(x)
    bimg = bimg*60-60
    bimg = bimg.detach().cpu().numpy().squeeze()

    return bimg



def fill_maskP(mask, bbox, maskP, bbox2):
    # Extract the dimensions of mask and maskP
    P, Q = mask.shape
    M, N = maskP.shape

    # Calculate the scaling factors for converting pixel positions to physical positions
    x_scale = (bbox[1] - bbox[0]) / Q
    z_scale = (bbox[3] - bbox[2]) / P

    xP_scale = (bbox2[1] - bbox2[0]) / N
    zP_scale = (bbox2[3] - bbox2[2]) / M

    # Iterate over each pixel in mask
    for p in range(P):
        for q in range(Q):
            # Check if the pixel value is True in mask
            if mask[p, q]:
                # Convert the pixel position to physical position in maskP
                x = bbox[0] + q * x_scale
                z = bbox[2] + p * z_scale

                # Calculate the corresponding pixel position in maskP
                xP = int((x - bbox2[0]) / xP_scale)
                zP = int((z - bbox2[2]) / zP_scale)

                # Check if the calculated pixel position is within the dimensions of maskP
                if 0 <= xP < N and 0 <= zP < M:
                    # Fill the corresponding position in maskP with True
                    maskP[zP, xP] = True

    return maskP


def make_bimg_goud(H, grid, model_dir, model_name, device, step, DGS_DOUBLE=False, cval=0.0):
    norm = np.maximum(np.max(np.abs(H.idata)), np.max(np.abs(H.qdata)))
    # norm = np.max(np.sqrt(H.idata ** 2 + H.qdata ** 2))
    H.idata /= norm
    H.qdata /= norm
    # P.idata = P.idata / np.amax(P.idata)
    # P.qdata = P.qdata / np.amax(P.qdata)
    npixels = grid.shape[0] * grid.shape[1]
    # ipixels_total = torch.randperm(npixels)
    ipixels_total = torch.arange(npixels)
    ipixels_chunks = ipixels_total.unfold(dimension=0, size=step, step=step)

    Y_cleanTotal = torch.zeros((npixels, 2))
    print("<GOUD")
    for (counter, ipixels) in enumerate(ipixels_chunks):
        print("GOUD: %d/%d" % (counter, len(ipixels_chunks)))
        model = load_goud_model(H, grid, ipixels, model_dir, model_name, device, cval, DGS_DOUBLE)
        raw_input = torch.concat((torch.from_numpy(H.idata), torch.from_numpy(H.qdata)), 0)
        raw_input = raw_input.to(device)
        y_clean, iqcum_goud = model(raw_input)

        y_clean_cpu = y_clean.detach().cpu()
        for idx, ipix in enumerate(ipixels):
            Y_cleanTotal[ipix, :] = y_clean_cpu[idx]
    print("\>GOUD")
    Y_cleanTotal_npy = Y_cleanTotal.numpy()
    iq = Y_cleanTotal_npy[:, 0] + 1j * Y_cleanTotal_npy[:, 1]
    iq = iq.reshape(grid.shape[:2])
    env = np.abs(iq)
    env = replace_zeros_numpy(env, cval)
    bimg_goud = 20 * np.log10(env/env.max())  # Log-compress
    # bimg_max = np.amax(bimg_goud)
    # bimg_goud = bimg_goud - bimg_max  # Normalize by max value

    return bimg_goud, Y_cleanTotal_npy


def make_goud_recon(channel_data, grid, model_dir, model_name, device, DGS_DOUBLE=False, cval=0.0):
    norm = np.maximum(channel_data)
    # norm = np.max(np.sqrt(H.idata ** 2 + H.qdata ** 2))
    channel_data /= norm
    npixels = grid.shape[0] * grid.shape[1]

    ipixels_total = torch.arange(npixels)
    ipixels_chunks = ipixels_total.unfold(dimension=0, size=300, step=300)

    Y_cleanTotal = torch.zeros((npixels, 2))
    for (counter, ipixels) in enumerate(ipixels_chunks):
        model = load_goud_model(H, grid, ipixels, model_dir, model_name, device, cval)
        raw_input = torch.concat((torch.from_numpy(H.idata), torch.from_numpy(H.qdata)), 0)
        raw_input = raw_input.to(device)
        y_clean, iqcum_goud = model(raw_input)

        y_clean_cpu = y_clean.detach().cpu()
        for idx, ipix in enumerate(ipixels):
            Y_cleanTotal[ipix, :] = y_clean_cpu[idx]
    # print("\>GOUD")
    Y_cleanTotal_npy = Y_cleanTotal.numpy()
    iq = Y_cleanTotal_npy[:, 0] + 1j * Y_cleanTotal_npy[:, 1]
    iq = iq.reshape(grid.shape[:2])
    env = np.abs(iq)
    env = replace_zeros_numpy(env, cval)
    bimg_goud = 20 * np.log10(env/env.max())  # Log-compress
    # bimg_max = np.amax(bimg_goud)
    # bimg_goud = bimg_goud - bimg_max  # Normalize by max value

    return bimg_goud, env


def mask_in_correct_dimension(grid_full, grid_roi, mask_roi, device):
    M, N, _ = grid_full.shape
    nrows_roi, ncols_roi = mask_roi.shape
    Lx_roi = grid_roi[0, 1, 0] - grid_roi[0, 0, 0]
    Lz_roi = grid_roi[1, 0, 2] - grid_roi[0, 0, 2]
    Lx_full = grid_full[0, 1, 0] - grid_full[0, 0, 0]
    Lz_full = grid_full[1, 0, 2] - grid_full[0, 0, 2]

    mask_full = np.zeros((M, N)).astype(bool)
    for ii in range(nrows_roi):
        for jj in range(ncols_roi):
            if mask_roi[ii, jj]:
                xc_roi = grid_roi[ii, jj, 0]
                zc_roi = grid_roi[ii, jj, 2]
                k = 1.5
                mask1 = (xc_roi - k*Lx_full) <= grid_full[:, :, 0]
                mask2 = grid_full[:, :, 0] <= (xc_roi + k*Lx_full)
                mask3 = (zc_roi - k*Lz_full) <= grid_full[:, :, 2]
                mask4 = grid_full[:, :, 2] <= (zc_roi + k*Lz_full)
                aux_mask = np.logical_and.reduce((mask1, mask2, mask3, mask4))
                mask_full = np.logical_or.reduce((mask_full, aux_mask))
    mask_torch = (1.0 * torch.flatten(torch.from_numpy(mask_full))[None, None, :]).to(device)
    return mask_full, mask_torch


def make_bimg_roth(H, model, grid, device):
    laterals = grid[0, :, 0]
    depths = grid[:, 0, 2]

    # norm = np.maximum(np.max(np.abs(P.idata)), np.max(np.abs(P.qdata)))
    norm = np.max(np.sqrt(H.idata ** 2 + H.qdata ** 2))

    H.idata /= norm
    H.qdata /= norm

    iq = torch.concat((torch.from_numpy(H.idata), torch.from_numpy(H.qdata)), 0)
    iq = iq.to(device)

    # print('<ROTH')
    singleWeight, td_input = model(iq)
    singleWeight = singleWeight.detach()
    td_input = td_input.detach()
    # print("ROTH\>")

    env_complex = torch.mean(td_input * singleWeight.flatten()[None, None, :, None].to(device), dim=-1)

    env = (torch.sqrt(torch.sum(env_complex.squeeze() ** 2, dim=0)) + 1e-6).cpu().numpy()
    env = env.reshape((len(depths), len(laterals)))
    bimg_roth = 20 * np.log10(env)
    bimg_max = np.amax(bimg_roth)
    bimg_roth = bimg_roth - bimg_max  # Normalize by max value

    return bimg_roth, env_complex, singleWeight, singleWeight.cpu().numpy().reshape((len(depths), len(laterals)))
    # return bimg_roth, env_complex, td_input, singleWeight.cpu().numpy().reshape((len(depths), len(laterals)))


def random_number_between_a(x, a):
    # Generate a random value between 0 and 1
    random_value = random.random()
    # Scale and shift the random value to fit the range [x - a, x + a]
    random_number = (2 * a * random_value) + (x - a)
    return random_number


def random_number_between(lim1, lim2):
    a = (lim2-lim1)/2
    x = (lim2+lim1)/2
    # Generate a random value between 0 and 1
    random_value = random.random()
    # Scale and shift the random value to fit the range [x - a, x + a]
    random_number = (2 * a * random_value) + (x - a)
    return random_number


def generate_2d_gaussian_kernel(kernel_size, sigma_x, sigma_y):
    # Create meshgrid
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(x, y)

    # Calculate Gaussian kernel
    kernel = np.exp(-0.5 * (xx**2 / sigma_x**2 + yy**2 / sigma_y**2))
    kernel /= (2 * np.pi * sigma_x * sigma_y)  # Normalize the kernel

    return kernel


def gaussian_random_mask(grid, ring_mask, min_len, max_len, threshold=0.0):
    random_matrix = np.random.normal(loc=0, scale=1, size=(grid.shape[0], grid.shape[1]))
    sigma_relation = 1/(grid.shape[0]/grid.shape[1])
    sigma_y = 12.5
    kernel = generate_2d_gaussian_kernel(kernel_size=25, sigma_x=sigma_relation*sigma_y, sigma_y=sigma_y)
    convolved_matrix = convolve2d(random_matrix, kernel, mode='same')
    # mask = convolved_matrix > 0.08
    # mask = convolved_matrix > 0.035
    mask = convolved_matrix > threshold
    mask = mask*ring_mask

    # min_mask_len = int(0.4*len(np.where(ring_mask)[0]))
    min_mask_len = min_len
    max_mask_len = max_len
    mask_len = len(np.where(mask)[0])

    print("\tSearching mask with %d < len < %d: \tWith threshold: %.4f,\t mask len: %d" % (min_mask_len,
                                                                                           max_mask_len,
                                                                                           threshold,
                                                                                           mask_len))
    if min_mask_len > mask_len:
        return gaussian_random_mask(grid, ring_mask, threshold=threshold-0.0001,
                                    min_len=min_len, max_len=max_len)
    elif min_mask_len <= mask_len <= max_mask_len:
        return mask
    else:
        return gaussian_random_mask(grid, ring_mask, threshold=threshold+0.0001,
                                    min_len=min_len, max_len=max_len)


def create_mask(grid, roi_ring, DEVICE, min_len=800, max_len=980):
    mask = gaussian_random_mask(grid, ring_mask=roi_ring, min_len=min_len, max_len=max_len)
    mask_torch = (1.0 * torch.flatten(torch.from_numpy(mask))[None, None, :]).to(DEVICE)
    mask_indices = torch.nonzero(mask_torch == 1)[:, 2]

    return mask, mask_torch, mask_indices


def grid_square_roi(H, L, nLaterals_full, nDepths_full, nair_case=False):
    # Extract position and radius
    xctr = H.pos_lat
    zctr = H.pos_ax

    # Get physical limit dimensions
    min_zlim, max_zlim = H.grid_zlims   # in mm
    min_xlim, max_xlim = H.grid_xlims  # in mm

    # Calculate ROI limits
    roi_min_zlim, roi_max_zlim = zctr - L/2, zctr + L/2
    roi_min_xlim, roi_max_xlim = xctr - L/2, xctr + L/2

    # Adjust ROI limits if they exceed physical limits
    roi_min_zlim = max(roi_min_zlim, min_zlim)
    roi_max_zlim = min(roi_max_zlim, max_zlim)
    roi_min_xlim = max(roi_min_xlim, min_xlim)
    roi_max_xlim = min(roi_max_xlim, max_xlim)

    # Update P
    H.roi_xlims = [roi_min_xlim, roi_max_xlim]
    H.roi_zlims = [roi_min_zlim, roi_max_zlim]

    # if nair_case:
    #     depths = np.linspace(roi_min_zlim, roi_max_zlim, num=800)
    #     # print(len(depths))
    #     input_Id, _ = create_input_Id(H, npoints=len(depths), depths=depths)
    #     # print(input_Id.shape)
    #     # input_Id.shape = (800, 128, 2)
    #     bbox_orig = (min_xlim, max_xlim, roi_min_zlim, roi_max_zlim)
    #     bbox_new = (roi_min_xlim, roi_max_xlim, roi_min_zlim, roi_max_zlim)
    #
    #     # print(bbox_orig)
    #     # print(bbox_new)
    #     #
    #     # print(input_Id[:, :, 0].T.shape)
    #     H.idata = interpolate_data(input_Id[:, :, 0].T, bbox_orig=bbox_orig, bbox_new=bbox_new)
    #     # input_Id[:, : 0].T.sahpe() = (128, 800)
    #     # H.idata.shape = (1, 128, 800)
    #     H.qdata = interpolate_data(input_Id[:, :, 1].T, bbox_orig=bbox_orig, bbox_new=bbox_new)

    # nLaterals = len(laterals)
    # nDepths = len(depths)
    nLaterals = np.round((H.roi_xlims[1] - H.roi_xlims[0]) * nLaterals_full / (H.grid_xlims[1] - H.grid_xlims[0])).astype('int')
    nDepths = np.round((H.roi_zlims[1] - H.roi_zlims[0]) * nDepths_full / (H.grid_zlims[1] - H.grid_zlims[0])).astype('int')

    print("#pixels in full: %d x %d" % (nLaterals_full, nDepths_full))
    print("#pixels in roi: %d x %d" % (nLaterals, nDepths))
    laterals = np.linspace(roi_min_xlim, roi_max_xlim, nLaterals)
    depths = np.linspace(roi_min_zlim, roi_max_zlim, nDepths)
    grid = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)
    # grid: [nrows, ncols, 3]
    # grid: [200, 32, 3]
    return H, grid


class myLoss(nn.Module):
    def __init__(self, weight=None, mask=None, max=None, cval=None):
        super(myLoss, self).__init__()
        self.MSE = nn.MSELoss()
        self.KL = nn.KLDivLoss()
        self.L1Loss = nn.L1Loss()
        self.weight = weight
        self.mask = mask
        self.max = max
        self.cval = cval

    def y2bimg(self, Y):
        Y = Y.clone()
        iq = Y[:, 0] ** 2 + Y[:, 1] ** 2
        env = torch.sqrt(iq)
        # env = torch.add(env, self.cval)
        env = torch.where(env != 0, env, torch.tensor(self.cval))

        # env = replace_zeros_torch(env, small_value=self.cval)
        bimg = 20 * torch.log10(env)  # Log-compress
        bimg = torch.add(bimg, -self.max)
        bimg = torch.clamp(bimg, min=-60, max=0) / 60
        return bimg

    def forward(self, Y, IQ, Yt, IQt):
        bimg = self.y2bimg(Y)
        # bimgt = self.y2bimg(Yt)
        # Y_Loss = -self.MSE(bimg, -torch.ones_like(bimg))

        iq = torch.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)

        min_val = torch.min(iq)
        max_val = torch.max(iq)
        iq = torch.add(iq, -min_val)
        iq = torch.mul(iq, 1/(max_val-min_val))

        Y_Loss = -self.MSE(iq, torch.zeros_like(iq))
        # Y_Loss = -self.MSE(iq, torch.ones_like(iq))
        # Y_Loss = -self.MSE(iq, torch.ones_like(iq))

        IQ_Loss = self.weight*self.MSE(IQ, IQt)
        total_loss = Y_Loss - IQ_Loss
        print("\tbimgLoss: %f\tiqLoss: %f" % (Y_Loss.item(), IQ_Loss.item()))

        return total_loss


class RothLoss(nn.Module):
    def __init__(self, mask=None):
        super(RothLoss, self).__init__()
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.KL = nn.KLDivLoss()
        self.mask = mask

    def forward(self, idx, weight_output, td_output, weight_target, td_target):
        # y_weight_masked = torch.masked_select(weight_output, self.mask.squeeze().reshape(t_weight.shape).bool())
        # t_weight_masked = torch.masked_select(weight_target, self.mask.squeeze().reshape(t_weight.shape).bool())

        mask = self.mask.squeeze().reshape(weight_output.shape).bool()
        weight_output_in = torch.masked_select(weight_output, ~mask)
        weight_output_out = torch.masked_select(weight_output, mask)

        # # Make both
        # size = weight_output_masked_out.shape
        # reshaped_tensor = weight_output_masked_in.view(1, 1, -1)
        # interpolated_tensor = F.interpolate(reshaped_tensor, size=size, mode='linear')
        # weight_output_masked_in = interpolated_tensor.view(-1)

        # print(weight_output_masked_in.shape)
        # print(weight_output_masked_out.shape)
        # t_weight_masked = torch.masked_select(t_weight, self.mask.squeeze().reshape(t_weight.shape).bool())

        weight_obj = -self.MSE(weight_output_out, torch.zeros_like(weight_output_out))
        td_obj = -self.MSE(td_output, td_target)
        # weight_loss = self.MSE(weight_output_masked_out, 1e-3*weight_output_masked_out)
        # weight_loss = self.KL(weight_output_masked_out, 1e-6*weight_output_masked_out)
        total_loss = weight_obj + td_obj
        print("\t%d.\ttotal: %f\tweight_loss: %f\ttd_loss: %f" % (idx,
                                                                total_loss.item(),
                                                                weight_obj.item(),
                                                                td_obj.item()))

        # weight_in = -self.KL(weight_output_masked_in, torch.ones_like(weight_output_masked_in))
        # weight_out = -self.KL(weight_output_masked_out, 1e-3*weight_output_masked_out)
        # # td_loss = 1e2*self.MSE(td_output, td_target)     # always positive
        #
        # total_loss = weight_in + weight_out
        # print("\t%d.\ttotal: %f - \tweight_in: %f\tweight_out: %f" % (idx,
        #                                                      total_loss.item(),
        #                                                      weight_in.item(),
        #                                                      weight_out.item()))

        # total_loss = weight_loss - td_loss
        # print("\t%d.\ttotal: %f\tweight_loss: %f\ttd_loss: %f" % (idx,
        #                                                         total_loss.item(),
        #                                                         weight_loss.item(),
        #                                                         td_loss.item()))
        return total_loss


def attack_goud2020onepixel(model, X, Yt, IQt, epsilon, alpha, attack_iters, lower_limit, upper_limit, p='inf', device='cpu', loss_fn=None, delta=None):

    delta = torch.zeros_like(X).to(device)
    delta.uniform_(-epsilon, epsilon)
    # delta.normal_(mean=0, std=1)
    X_adv = torch.zeros_like(X).to(device)
    X_adv.requires_grad_(True)
    X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
    delta.data = X_adv - X

    for idx, _ in enumerate(range(attack_iters)):
        # print('.', end='')
        X_adv.data = X_adv.data / torch.max(torch.max(torch.abs(X_adv.data[0, :, :]),
                                                      torch.max(torch.abs(X_adv.data[1, :, :]))))
        Y, IQ = model(X_adv)

        # loss = loss_fn(Y, torch.zeros_like(Y))
        loss = loss_fn(Y, IQ, Yt, IQt)
        # print("%i: %f" % (idx, loss.item()))
        loss.backward(retain_graph=True)

        grad = X_adv.grad.detach()
        if p == 'inf':
            delta.data = torch.clamp(delta + alpha * grad.sign(), min=-epsilon, max=epsilon)
        else:
            raise NotImplementedError
        X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
        delta.data = X_adv - X
        X_adv.grad.zero_()
    # print('|', end='\n')
    X_adv.data = X_adv.data / torch.max(torch.max(torch.abs(X_adv.data[0, :, :]),
                                                  torch.max(torch.abs(X_adv.data[1, :, :]))))
    return delta.detach(), X_adv.detach()


def attack_roth(model, X, weight_target, td_target, epsilon, alpha, attack_iters,
                lower_limit, upper_limit, p='inf', device='cpu', loss_fn=None):
    delta = torch.zeros_like(X).to(device)
    delta.uniform_(-epsilon, epsilon)
    X_adv = torch.zeros_like(X).to(device)
    X_adv.requires_grad_(True)
    X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
    delta.data = X_adv - X

    for idx, _ in enumerate(range(attack_iters)):
        print('.', end='')
        # X_adv.data = X_adv.data/torch.max(torch.sqrt(X_adv.data[0, :, :] ** 2 + X_adv.data[1, :, :] ** 2))

        weight_output, td_output = model(X_adv)

        loss = loss_fn(idx, weight_output, td_output, weight_target, td_target)
        loss.backward(retain_graph=True)

        grad = X_adv.grad.detach()

        if p == 'inf':
            delta.data = torch.clamp(delta + alpha * grad.sign(), min=-epsilon, max=epsilon)
        else:
            raise NotImplementedError
        X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
        delta.data = X_adv - X
        X_adv.grad.zero_()
    print('|', end='\n')
    # X_adv.data = X_adv.data / torch.max(torch.sqrt(X_adv.data[0, :, :] ** 2 + X_adv.data[1, :, :] ** 2))
    return delta.detach(), X_adv.detach()


def attack_nair2020(model, X, epsilon, alpha, attack_iters, lower_limit, upper_limit, p='inf', device='cpu', loss_fn=None):
    model.eval()
    X = X.to(device)

    delta = torch.zeros_like(X).to(device)
    delta.uniform_(-epsilon, epsilon)
    X_adv = torch.zeros_like(X).to(device)
    X_adv.requires_grad_(True)
    X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
    delta.data = X_adv - X

    for idx, _ in enumerate(range(attack_iters)):
        outputs_beam, outputs_seg = model(X_adv)
        loss = loss_fn(outputs_beam)
        loss.backward(retain_graph=True)

        grad = X_adv.grad.detach()

        if p == 'inf':
            delta.data = torch.clamp(delta + alpha * grad.sign(), min=-epsilon, max=epsilon)
        else:
            raise NotImplementedError
        X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
        delta.data = X_adv - X
        X_adv.grad.zero_()
    # print('')
    return delta.detach(), X_adv.detach()


def attack_unet1deco(model, X, epsilon, alpha, attack_iters, lower_limit, upper_limit, p='inf', device='cpu', loss_fn=None):
    model.eval()
    X = X.to(device)

    delta = torch.zeros_like(X).to(device)
    delta.uniform_(-epsilon, epsilon)
    X_adv = torch.zeros_like(X).to(device)
    X_adv.requires_grad_(True)
    X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
    delta.data = X_adv - X

    for idx, _ in enumerate(range(attack_iters)):
        outputs_beam = model(X_adv)
        loss = loss_fn(outputs_beam)
        loss.backward(retain_graph=True)

        grad = X_adv.grad.detach()

        if p == 'inf':
            delta.data = torch.clamp(delta + alpha * grad.sign(), min=-epsilon, max=epsilon)
        else:
            raise NotImplementedError
        X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
        delta.data = X_adv - X
        X_adv.grad.zero_()
    # print('')
    return delta.detach(), X_adv.detach()


def p2tensor(H, device):
    x = torch.concat((torch.from_numpy(H.idata), torch.from_numpy(H.qdata)), 0)
    x = x.to(device)
    return x


def replace_zeros_torch(tensor, small_value):
    mask = (tensor == 0)
    tensor[mask] = small_value
    return tensor

def replace_zeros_numpy(array, small_value):
    array[array == 0] = small_value
    return array


maxmin_norm = lambda x: (x - x.min()) / (x.max() - x.min())
# ms_ssim = MS_SSIM(data_range=60.0, size_average=True, channel=1, win_size=7, K=(0.01, 0.03))
ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1, win_size=7, K=(0.01, 0.03))
def compute_metrics(bmode, bmode_ref, roi_in, roi_out, roi_cyst, threshold, device=None):
    # Compute metrics
    dct = {}
    env = 10**(bmode/20)
    env_in = env[roi_in]
    env_out = env[roi_out]
    dct['contrast'] = contrast(env_in, env_out)
    dct['cnr'] = cnr(env_in, env_out)
    dct['gcnr'] = gcnr(env_in, env_out)
    dct['snr'] = snr(env_out)
    dct['psnr'] = psnr(bmode, bmode_ref)

    bmode_img = (bmode + 60) / 60
    bmode_nl = denoise_nl_means(bmode_img,
                                h=0.3,
                                patch_size=5,
                                patch_distance=9,
                                preserve_range=True)
    # if case == 'strohm':
    #     bmode_nl = denoise_nl_means(bmode_img,
    #                                 h=0.3,
    #                                 patch_size=5,
    #                                 patch_distance=9,
    #                                 preserve_range=True)
    # else:
    #     bmode_nl = denoise_nl_means(bmode_img,
    #                                 h=0.1,
    #                                 patch_size=5,
    #                                 patch_distance=9,
    #                                 preserve_range=True)
    bmode_nl = maxmin_norm(bmode_nl)
    roi_comp = bmode_nl<threshold
    msssim_ref = torch.from_numpy((roi_cyst[None, None, :]*1.0).astype('float32'))
    msssim_comp = torch.from_numpy((roi_comp[None, None, :]*1.0).astype('float32'))
    dct['ms_ssim'] = ms_ssim(msssim_ref, msssim_comp).item()

    return dct, bmode_nl


def compute_metrics_old(env, bmode, bmodeRef, roi_cyst, roi_ring):
    # Compute metrics
    env = 10**(bmode/20)
    env_cyst = env[roi_cyst]
    env_ring = env[roi_ring]
    contrast_value = contrast(env_cyst, env_ring)
    snr_value = snr(env_ring)
    gcnr_value = gcnr(env_cyst, env_ring)
    cnr_value = cnr(env_cyst, env_ring)
    psnr_value = psnr(bmode, bmodeRef)
    return contrast_value, cnr_value, gcnr_value, snr_value, psnr_value


def interpolate_data(input_data, bbox_orig, bbox_new, size_output):
    x1_orig, x2_orig, z1_orig, z2_orig = bbox_orig
    x1_new, x2_new, z1_new, z2_new = bbox_new

    # Define the x-axis and z-axis positions of the input matrix
    x_axis = np.linspace(x1_orig, x2_orig, input_data.shape[1])
    z_axis = np.linspace(z1_orig, z2_orig, input_data.shape[0])


    # Create the interpolation function using the input data and axis positions
    interpolator = RectBivariateSpline(z_axis, x_axis, input_data)

    # Define the x-axis and z-axis positions of the output matrix
    output_x_axis = np.linspace(x1_new, x2_new, size_output[0])
    output_z_axis = np.linspace(z1_new, z2_new, size_output[1])

    # Interpolate the data using the output axis positions
    output_data = interpolator(output_z_axis, output_x_axis)

    return output_data


class NairLoss(nn.Module):
    def __init__(self, mask=None):
        super(NairLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.mask = mask

    def forward(self, output_beam):
    # def forward(self, output_beam, output_seg, target_enh, target_seg, smooth=1):
        loss = -self.MSE(torch.flatten(output_beam), 1-torch.flatten(self.mask))
        print("\tLoss: %f" % loss)
        # # flatten label and prediction tensors
        # output_beam = output_beam.view(-1)
        # output_seg = output_seg.view(-1)
        # target_enh = target_enh.view(-1)
        # target_seg = target_seg.view(-1)
        #
        # intersection = (output_seg * target_seg).sum()
        # dice_loss = 1 - (2. * intersection + smooth) / (output_seg.sum() + target_seg.sum() + smooth)
        #
        # return self.l1_loss(output_beam, target_enh) + dice_loss

        return loss


def nair_input_2(H, device):
    H.idata = H.idata / np.amax(H.idata)
    H.qdata = H.qdata / np.amax(H.qdata)
    depths = np.linspace(30, 80, num=800) / 1000
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    x[:, :, 0] = x[:, :, 0] / np.amax(x[:, :, 0])
    x[:, :, 1] = x[:, :, 1] / np.amax(x[:, :, 1])
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)
    # x: 1, 2, npoints, 128
    return x, depth_samples, tzero_samples


def nair_input(H, device):
    norm_value = np.max((np.abs(H.idata), np.abs(H.qdata)))
    print(f'nair_input_corrected: -> norm_value: {norm_value}')
    H.idata = H.idata / norm_value
    H.qdata = H.qdata / norm_value

    depths = np.linspace(30, 80, num=800) / 1000
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    norm_value_x = np.max(np.abs(x))
    print(f'nair_input_corrected: -> norm_value_x: {norm_value_x}')

    x = x/norm_value_x
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)
    # x: 1, 2, npoints, 128
    return x, depth_samples, tzero_samples

def strohm_input(H, device):
    H.idata = H.idata / np.amax(H.idata)
    H.qdata = H.qdata / np.amax(H.qdata)
    depths = np.linspace(30, 80, num=1600) / 1000
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    x[:, :, 0] = x[:, :, 0] / np.amax(x[:, :, 0])
    x[:, :, 1] = x[:, :, 1] / np.amax(x[:, :, 1])
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)
    # x: 1, 2, npoints, 128
    return x, depth_samples, tzero_samples

def wang_input(H, device):
    H.idata = H.idata / np.amax(H.idata)
    H.qdata = H.qdata / np.amax(H.qdata)
    depths = np.linspace(30, 80, num=800) / 1000
    x, depth_samples, tzero_samples = create_input_Id(H, npoints=len(depths), depths=depths)  # x: npointsx128x2
    x[:, :, 0] = x[:, :, 0] / np.amax(x[:, :, 0])
    x[:, :, 1] = x[:, :, 1] / np.amax(x[:, :, 1])
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)
    # x: 1, 2, npoints, 128
    return x, depth_samples, tzero_samples

def downsample_channel_data(H, laterals, depths, device):
    norm_value = np.max((np.abs(H.idata), np.abs(H.qdata)))
    H.idata = H.idata / norm_value
    H.qdata = H.qdata / norm_value
    # 0.002, 0.052
    # x, depth_samples, tzero_samples = create_input_Id(H,npoints, depths)  # x: npointsx128x2
    x = roi_channel_data(H,laterals=laterals, depths=depths)  # x: npointsx128x2
    # norm_value_x = np.max(np.abs(x))
    # print(f'nair_input_corrected: -> norm_value_x: {norm_value_x}')

    # x = x/norm_value_x
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)
    # x: 1, 2, npoints, 128
    return x


def create_input_Id(P, npoints, depths):
    depths = np.linspace(depths[0], depths[-1], num=npoints)
    depth_samples = 2 * (depths / P.c) * P.fs
    tzero_samples = int(P.time_zero * P.fs)

    input_Id = np.zeros((npoints, 128, 2))
    for idx in np.arange(P.idata.shape[1]):
        iplane = np.concatenate((np.zeros(tzero_samples), P.idata[0, idx, :]))  # fill with zeros
        qplane = np.concatenate((np.zeros(tzero_samples), P.qdata[0, idx, :]))  # fill with zeros

        data_axis = np.arange(len(iplane))

        func_i = interpolate.interp1d(data_axis, iplane, kind='slinear')
        func_q = interpolate.interp1d(data_axis, qplane, kind='slinear')

        input_Id[:, idx, 0] = func_i(depth_samples)
        input_Id[:, idx, 1] = func_q(depth_samples)

    # iq = np.sqrt(input_Id[:, :, 0] ** 2 + input_Id[:, :, 1] ** 2)
    return input_Id, depth_samples, tzero_samples

def roi_channel_data(P, laterals, depths):

    nLats, nDepths = len(laterals), len(depths)
    depth_samples = 2 * (depths / P.c) * P.fs
    tzero_samples = int(P.time_zero * P.fs)

    donwsample_depths = np.zeros((nDepths, P.idata.shape[1], 2))
    for idx in np.arange(P.idata.shape[1]):
        iplane = np.concatenate((np.zeros(tzero_samples), P.idata[0, idx, :]))  # fill with zeros
        qplane = np.concatenate((np.zeros(tzero_samples), P.qdata[0, idx, :]))  # fill with zeros

        data_axis = np.arange(len(iplane))

        func_i = interpolate.interp1d(data_axis, iplane, kind='slinear')
        func_q = interpolate.interp1d(data_axis, qplane, kind='slinear')

        donwsample_depths[:, idx, 0] = func_i(depth_samples)
        donwsample_depths[:, idx, 1] = func_q(depth_samples)

    XDC_elements_laterals = np.linspace(P.phantom_xlims[0], P.phantom_xlims[-1], P.idata.shape[1])
    donwsample_total = np.zeros((nDepths, nLats, 2))
    for idx in np.arange(nDepths):
        iplane = donwsample_depths[idx, :, 0]
        qplane = donwsample_depths[idx, :, 1]

        func_i = interpolate.interp1d(XDC_elements_laterals, iplane, kind='slinear')
        func_q = interpolate.interp1d(XDC_elements_laterals, qplane, kind='slinear')

        donwsample_total[idx, :, 0] = func_i(laterals)
        donwsample_total[idx, :, 1] = func_q(laterals)

    # iq = np.sqrt(input_Id[:, :, 0] ** 2 + input_Id[:, :, 1] ** 2)
    return donwsample_total


def nair2raw(H, nair_input_adv_npy, depth_values, offset, device):
    nTransducers = H.idata.shape[1]
    nSamples = H.idata.shape[2]
    depth_sample_min = int(depth_values.min())+1
    depth_sample_max = int(depth_values.max())

    total_depth = np.concatenate([depth_values, np.arange(nSamples)-offset])
    for idx in range(nTransducers):
        total_i = np.concatenate([nair_input_adv_npy[0, 0, :, idx], H.idata[0, idx, :]])
        total_q = np.concatenate([nair_input_adv_npy[0, 1, :, idx], H.qdata[0, idx, :]])
        func_i = interpolate.interp1d(total_depth, total_i, kind='slinear', assume_sorted=False)
        func_q = interpolate.interp1d(total_depth, total_q, kind='slinear', assume_sorted=False)

        # func_i = interpolate.interp1d(depth_values, input_adv_npy[0, 0, :, idx], kind='slinear')
        # func_q = interpolate.interp1d(depth_values, input_adv_npy[0, 1, :, idx], kind='slinear')
        for depth_sample in range(depth_sample_min, depth_sample_max):
            sample_num = depth_sample - offset
            H.idata[0, idx, sample_num] = func_i(depth_sample)
            H.qdata[0, idx, sample_num] = func_q(depth_sample)

    # for counter, depth_value in enumerate(depth_values):
    #     if 1 <= counter < len(depth_values)-1:
    #         depth_sample_n = int(depth_value)+1
    #         x = np.array([depth_values[counter-1], depth_sample_n - 1, depth_value, depth_sample_n + 1, depth_values[counter+1]])
    #         # x = np.array([depth_values[counter-1], depth_sample_n - 1, depth_value, depth_sample_n + 1, depth_values[counter+1]])
    #         for idx in range(nTransducers):
    #             sample_num = depth_sample_n-offset
    #             y_i = np.array([
    #                 input_adv_npy[0, 0, counter-1, idx],
    #                 H.idata[0, idx, sample_num-1],
    #                 input_adv_npy[0, 0, counter, idx],
    #                 H.idata[0, idx, sample_num+1],
    #                 input_adv_npy[0, 0, counter+1, idx],
    #             ])
    #
    #             y_q = np.array([
    #                 input_adv_npy[0, 1, counter-1, idx],
    #                 H.qdata[0, idx, sample_num-1],
    #                 input_adv_npy[0, 1, counter, idx],
    #                 H.qdata[0, idx, sample_num+1],
    #                 input_adv_npy[0, 1, counter+1, idx],
    #             ])
    #
    #             print(x)
    #             print(y_i)
    #             print(y_q)
    #             func_i = interpolate.interp1d(x, y_i, kind='slinear', assume_sorted=False)
    #             func_q = interpolate.interp1d(x, y_q, kind='slinear', assume_sorted=False)
    #
    #             H.idata[0, idx, sample_num] = func_i(depth_sample_n)
    #             H.qdata[0, idx, sample_num] = func_q(depth_sample_n)

    # for idx in range(nTransducers):
    #     func_i = interpolate.interp1d(depth_values, input_adv_npy[0, 0, :, idx], kind='slinear')
    #     func_q = interpolate.interp1d(depth_values, input_adv_npy[0, 1, :, idx], kind='slinear')
    #     for depth_sample in range(depth_sample_min, depth_sample_max):
    #         sample_num = depth_sample - tzero_samples
    #         H.idata[0, idx, sample_num] = func_i(depth_sample)
    #         H.qdata[0, idx, sample_num] = func_q(depth_sample)
    raw_input = p2tensor(H, device)
    return H, raw_input.cpu().numpy()


def delta2raw(H, delta_npy, depth_samples, offset, device):
    nTransducers = H.idata.shape[1]

    for idx in range(nTransducers):
        deltas_i = delta_npy[0, 0, :, idx]
        deltas_q = delta_npy[0, 1, :, idx]

        samples_inf = np.floor(depth_samples).astype('int') - offset
        # samples_sup = samples_inf + 1

        # weights_inf = 1 - (depth_samples - np.floor(depth_samples))
        # weights_sup = 1 - weights_inf

        H.idata[0, idx, samples_inf] += deltas_i
        # H.idata[0, idx, samples_sup] += deltas_i

        H.qdata[0, idx, samples_inf] += deltas_q
        # H.qdata[0, idx, samples_sup] += deltas_q

        # H.idata[0, idx, samples_inf] += weights_inf*deltas_i
        # H.idata[0, idx, samples_sup] += weights_sup*deltas_i
        #
        # H.qdata[0, idx, samples_inf] += weights_inf*deltas_q
        # H.qdata[0, idx, samples_sup] += weights_sup*deltas_q

        # for counter, depth_sample in enumerate(depth_samples):
        #     delta_i = delta_npy[0, 0, counter, idx]
        #     delta_q = delta_npy[0, 1, counter, idx]
        #     sample_inf = int(depth_sample) - offset
        #     sample_sup = sample_inf + 1
        #     H.idata[0, idx, sample_inf] = H.idata[0, idx, sample_inf] + delta_i
        #     H.idata[0, idx, sample_sup] = H.idata[0, idx, sample_sup] + delta_i
        #
        #     H.qdata[0, idx, sample_inf] = H.qdata[0, idx, sample_inf] + delta_q
        #     H.qdata[0, idx, sample_sup] = H.qdata[0, idx, sample_sup] + delta_q

    raw_input = p2tensor(H, device)

    return H, raw_input.cpu().numpy()


def load_simu_numpy_names(file_path):
    with open(file_path, 'r') as file:
        file_list = file.read().splitlines()
    return file_list

from scipy.signal import hilbert
from scipy import signal
from scipy.interpolate import interp1d

def makeAberration(P, fwhm, rms_strength, seed):
    Fs = P.fs
    pitch = (P.ele_pos[1, 0] - P.ele_pos[0, 0]) * 1000  # pitch in mm
    recorded_signals = P.idata.squeeze()
    num_elements, num_samples = recorded_signals.shape

    # Step 1: Create profile
    np.random.seed(seed)
    rand_arr = np.random.normal(loc=0, scale=1, size=(num_elements, 1))
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Standard deviation of the Gaussian function (in mm)
    sigma_in_px = sigma / pitch
    gauss_kernel = signal.gaussian(num_elements, std=sigma_in_px)[:, None]
    profile = signal.fftconvolve(rand_arr, gauss_kernel, mode='same')
    rms_value = np.sqrt(np.mean(profile ** 2, axis=0))
    profile = (1e-9)*rms_strength*profile/rms_value   # from ns to seconds
    # profile = (1e-9)*rms_strength*np.ones_like(profile)

    aberrated_signals = np.zeros_like(recorded_signals)
    start, end = 0, (num_samples/Fs)     # in seconds
    time = np.linspace(start, end, num_samples)        # in seconds
    for idx, delay in enumerate(profile):
        new_time = time+delay
        curr_signal = recorded_signals[idx, :]
        interpolated_signal = interp1d(time, curr_signal,
                                       kind='linear',
                                       fill_value='extrapolate')(new_time)
        aberrated_signals[idx, :] = interpolated_signal

    P.idata = aberrated_signals[None, :]
    P.qdata = np.imag(hilbert(P.idata, axis=-1))

    dct = {'gauss_kernel': gauss_kernel,
           'signals': recorded_signals,
           'aberrated_signals': aberrated_signals}

    # return P, phase_screen, np.mean(phase_screen, axis=1), dct
    return P, profile, dct


def create_run_description(directory, description):
    """
    Create a text file with the provided description inside the given directory.

    Args:
    - directory (str): The directory path where the description file will be created.
    - description (str): The description text to be written into the file.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path for the description file
    description_file_path = os.path.join(directory, 'run_description.txt')

    # Write the description to the file
    with open(description_file_path, 'w') as file:
        file.write(description)


def attack_model_during_training(model, case, X, targets, epsilon, alpha, attack_iters,
                                 lower_limit, upper_limit, p='inf', device='cpu', loss_fn=None):
    model.eval()
    X = X.to(device)

    delta = torch.zeros_like(X).to(device)
    delta.uniform_(-epsilon, epsilon)
    X_adv = torch.zeros_like(X).to(device)
    X_adv.requires_grad_(True)
    X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
    delta.data = X_adv - X

    for idx, _ in enumerate(range(attack_iters)):
        if case == 'wang':
            N, C, H, W = X.size()
            z = torch.randn(N, 1, H, W).to(device)
            outputs_beam = model(torch.cat((X_adv, z), dim=1))
        else:
            outputs_beam = model(X_adv)

        loss = loss_fn(outputs_beam, targets)
        loss.backward(retain_graph=True)

        grad = X_adv.grad.detach()

        if p == 'inf':
            delta.data = torch.clamp(delta + alpha * grad.sign(), min=-epsilon, max=epsilon)
        else:
            raise NotImplementedError
        X_adv.data = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
        delta.data = X_adv - X
        X_adv.grad.zero_()
    # print('')
    return delta.detach(), X_adv.detach()


