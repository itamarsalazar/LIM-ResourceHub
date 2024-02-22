import numpy as np
from scipy import interpolate


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