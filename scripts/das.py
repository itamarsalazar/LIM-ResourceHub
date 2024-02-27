import copy
import torch
import numpy as np
from torch.nn.functional import grid_sample

#### Code adapted from the CUBDL repo. ###

PI = 3.14159265359


## Simple phase rotation of I and Q component by complex angle theta
def complex_rotate(I, Q, theta):
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)
    return Ir, Qr


def delay_plane(grid, angles):
    # Use broadcasting to simplify computations
    x = grid[:, 0].unsqueeze(0)
    z = grid[:, 2].unsqueeze(0)
    # For each element, compute distance to pixels
    dist = x * torch.sin(angles) + z * torch.cos(angles)
    # Output has shape [nangles, npixels]
    return dist


def delay_focus(grid, ele_pos):
    # Compute distance to user-defined pixels from elements
    # Expects all inputs to be torch tensors specified in SI units.
    # grid    Pixel positions in x,y,z    [npixels, 3]
    # ele_pos Element positions in x,y,z  [nelems, 3]
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = torch.norm(grid - ele_pos.unsqueeze(0), dim=-1)
    # Output has shape [nelems, npixels]
    return dist


class DAS_PW(torch.nn.Module):
    def __init__(
        self,
        P,
        grid,
        ang_list=None,
        ele_list=None,
        rxfnum=2,
        dtype=torch.float,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """ Initialization method for DAS_PW.

        All inputs are specified in SI units, and stored in self as PyTorch tensors.
        INPUTS
        P           A PlaneWaveData object that describes the acquisition
        grid        A [ncols, nrows, 3] numpy array of the reconstruction grid
        ang_list    A list of the angles to use in the reconstruction
        ele_list    A list of the elements to use in the reconstruction
        rxfnum      The f-number to use for receive apodization
        dtype       The torch Tensor datatype (defaults to torch.float)
        device      The torch Tensor device (defaults to GPU execution)
        """
        super().__init__()
        # If no angle or element list is provided, delay-and-sum all
        if ang_list is None:
            ang_list = range(P.angles.shape[0])
        elif not hasattr(ang_list, "__getitem__"):
            ang_list = [ang_list]
        if ele_list is None:
            ele_list = range(P.ele_pos.shape[0])
        elif not hasattr(ele_list, "__getitem__"):
            ele_list = [ele_list]

        # Convert plane wave data to tensors
        self.angles = torch.tensor(P.angles, dtype=dtype, device=device)
        self.ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)
        self.fc = torch.tensor(P.fc, dtype=dtype, device=device)
        self.fs = torch.tensor(P.fs, dtype=dtype, device=device)
        self.fdemod = torch.tensor(P.fdemod, dtype=dtype, device=device)
        self.c = torch.tensor(P.c, dtype=dtype, device=device)
        self.time_zero = torch.tensor(P.time_zero, dtype=dtype, device=device)

        # Convert grid to tensor
        self.grid = torch.tensor(grid, dtype=dtype, device=device).reshape(-1, 3)
        self.out_shape = grid.shape[:-1]

        # Store other information as well
        self.ang_list = torch.tensor(ang_list, dtype=torch.long, device=device)
        self.ele_list = torch.tensor(ele_list, dtype=torch.long, device=device)
        self.dtype = dtype
        self.device = device

    def forward(self, x, accumulate=False):
        """ Forward pass for DAS_PW neural network. """
        dtype, device = self.dtype, self.device

        # Load data onto device as a torch tensor

        idata, qdata = x
        idata = torch.tensor(idata, dtype=dtype, device=device)
        qdata = torch.tensor(qdata, dtype=dtype, device=device)

        # Compute delays in meters
        nangles = len(self.ang_list)
        nelems = len(self.ele_list)
        npixels = self.grid.shape[0]
        xlims = (self.ele_pos[0, 0], self.ele_pos[-1, 0])  # Aperture width
        txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)
        rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)
        txapo = torch.ones((nangles, npixels), dtype=dtype, device=device)
        rxapo = torch.ones((nelems, npixels), dtype=dtype, device=device)
        for i, tx in enumerate(self.ang_list):
            txdel[i] = delay_plane(self.grid, self.angles[[tx]])
            # txdel[i] += self.time_zero[tx] * self.c   # ORIGINAL
            txdel[i] -= self.time_zero[tx] * self.c     # IT HAS TO BE "-"
            # txapo[i] = apod_plane(self.grid, self.angles[tx], xlims)
        for j, rx in enumerate(self.ele_list):
            rxdel[j] = delay_focus(self.grid, self.ele_pos[[rx]])
            # rxapo[i] = apod_focus(self.grid, self.ele_pos[rx])

        # Convert to samples
        txdel *= self.fs / self.c
        rxdel *= self.fs / self.c

        # Initialize the output array
        idas = torch.zeros(npixels, dtype=self.dtype, device=self.device)
        qdas = torch.zeros(npixels, dtype=self.dtype, device=self.device)
        iq_cum = None
        if accumulate:
            iq_cum = torch.zeros(nangles, npixels, nelems, 2, dtype=dtype, device='cpu')
        # Loop over angles and elements
        # for t, td, ta in zip(self.ang_list, txdel, txapo):
        # for idx1, (t, td, ta) in tqdm(enumerate(zip(self.ang_list, txdel, txapo)), total=nangles):
        for idx1, (t, td, ta) in enumerate(zip(self.ang_list, txdel, txapo)):
            for idx2, (r, rd, ra) in enumerate(zip(self.ele_list, rxdel, rxapo)):
                # Grab data from t-th Tx, r-th Rx
                # Avoiding stack because of autograd problems
                # iq = torch.stack((idata[t, r], qdata[t, r]), dim=0).view(1, 2, 1, -1)
                i_iq = idata[t, r].view(1, 1, 1, -1)
                q_iq = qdata[t, r].view(1, 1, 1, -1)
                # Convert delays to be used with grid_sample
                delays = td + rd
                dgs = (delays.view(1, 1, -1, 1) * 2 + 1) / idata.shape[-1] - 1
                dgs = torch.cat((dgs, 0 * dgs), axis=-1)
                # Interpolate using grid_sample and vectorize using view(-1)
                # ifoc, qfoc = grid_sample(iq, dgs, align_corners=False).view(2, -1)
                ifoc = grid_sample(i_iq, dgs, align_corners=False).view(-1)
                qfoc = grid_sample(q_iq, dgs, align_corners=False).view(-1)
                # torch.Size([144130])
                # Apply phase-rotation if focusing demodulated data
                if self.fdemod != 0:
                    tshift = delays.view(-1) / self.fs - self.grid[:, 2] * 2 / self.c
                    theta = 2 * PI * self.fdemod * tshift
                    ifoc, qfoc = complex_rotate(ifoc, qfoc, theta)
                # Apply apodization, reshape, and add to running sum
                # apods = ta * ra
                # idas += ifoc * apods
                # qdas += qfoc * apods
                idas += ifoc
                qdas += qfoc
                # torch.Size([355*406])
                if accumulate:
                    # 1, npixels, nelems, 2
                    iq_cum[idx1, :, idx2, 0] = ifoc.cpu()
                    iq_cum[idx1, :, idx2, 1] = qfoc.cpu()

        # Finally, restore the original pixel grid shape and convert to numpy array
        idas = idas.view(self.out_shape)
        qdas = qdas.view(self.out_shape)

        env = torch.sqrt(idas**2 + qdas**2)
        bimg = 20 * torch.log10(env + torch.tensor(1.0*1e-25))
        bimg = bimg - torch.max(bimg)
        return bimg, env, idas, qdas, iq_cum


def make_bimg_das1(P, grid, device):
    norm_value = np.max((np.abs(P.idata), np.abs(P.qdata)))
    P.idata = P.idata / norm_value
    P.qdata = P.qdata / norm_value
    # norm = np.max(np.sqrt(P.idata ** 2 + P.qdata ** 2))
    # P.idata = P.idata/norm
    # P.qdata = P.qdata/norm

    id_angle = len(P.angles) // 2
    dasNet = DAS_PW(P, grid, ang_list=id_angle, device=device)
    bimg, env, _, _, _ = dasNet((P.idata, P.qdata), accumulate=False)
    bimg = bimg.detach().cpu().numpy()
    env = env.detach().cpu().numpy()
    return bimg, env

def make_pixel_grid(xlims, zlims, dx, dz):
    x_pos = np.arange(xlims[0], xlims[1], dx)
    z_pos = np.arange(zlims[0], zlims[1], dz)
    zz, xx = np.meshgrid(z_pos, x_pos, indexing="ij") # 'ij' -> rows: z, columns: x
    yy = xx * 0
    grid = np.stack((xx, yy, zz), axis=-1)  # [nrows, ncols, 3]
    return grid

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    P = LoadData(orig='fieldII', acq='simulation',
                     target='contrast_speckle_01', modulation='rf', param='2MHz')

    wvln = P.c / P.fc
    dx, dz = wvln / 3, wvln / 3
    grid = make_pixel_grid(P.grid_xlims, P.grid_zlims, dx, dz)
    bmode, _ = make_bimg_das1(copy.deepcopy(P), grid, device=device)
