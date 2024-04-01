import numpy as np
import h5py
import os
from scipy.signal import hilbert
import src.GLOBAL_VARIABLES as GV


class PlaneWaveData:
    """ A template class that contains the plane wave data.

    PlaneWaveData is a container or dataclass that holds all of the information
    describing a plane wave acquisition. Users should create a subclass that reimplements
    __init__() according to how their data is stored.

    The required information is:
    idata       In-phase (real) data with shape (nangles, nchans, nsamps)
    qdata       Quadrature (imag) data with shape (nangles, nchans, nsamps)
    angles      List of angles [radians]
    ele_pos     Element positions with shape (N,3) [m]
    fc          Center frequency [Hz]
    fs          Sampling frequency [Hz]
    fdemod      Demodulation frequency, if data is demodulated [Hz]
    c           Speed of sound [m/s]
    time_zero   List of time zeroes for each acquisition [s]

    Correct implementation can be checked by using the validate() method. See the
    PICMUSData class for a fully implemented example.
    """

    def __init__(self):
        """ Users must re-implement this function to load their own data. """
        # Do not actually use PlaneWaveData.__init__() as is.
        raise NotImplementedError
        # We provide the following as a visual example for a __init__() method.
        nangles, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        self.idata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.qdata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.angles = np.zeros((nangles,), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nangles,), dtype="float32")

    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles
        # print("Dataset successfully loaded")


class LoadDataCUBDL(PlaneWaveData):
    def __init__(self, dataset_path, task, filename):
        # Load dataset
        self.dir = os.path.join(dataset_path, task)
        self.filename = filename
        self.fname = os.path.join(self.dir, self.filename)
        self.code = filename[0:3]

        if self.code == 'OSL':
            with h5py.File(self.fname, "r") as f:
                # f = g["US"]["US_DATASET0000"]
                self.bimg = np.array(f["beamformed_data"], dtype="float32")
                self.idata = np.array(f["channel_data"], dtype="float32")   # (nangles, nXDCs, nsamples)
                self.qdata = None
                self.nXDCs = np.array(f["element_count"]).item()
                self.ele_pos = np.array(f["element_positions"]).T   # (nxDCs, 3)
                self.fc = np.array(f['modulation_frequency']).item()
                self.pixel_positions = np.array(f['pixel_positions']) #(3, H, W)
                self.fs = np.array(f["sampling_frequency"]).item()
                self.c = np.array(f["sound_speed"]).item()
                self.time_zero = np.array(f["start_time"]).squeeze()    # (nagles,)
                self.nangles = np.array(f['transmit_count']).item()
                self.angles = np.array(f['transmit_direction'])[0]      # (nangles,)
                # grid info
                self.grid_xlims = [self.ele_pos[0, 0], self.ele_pos[-1, 0]]
                self.grid_zlims = None
                modulation = "rf"
        elif self.code == 'UFL':
            with h5py.File(self.fname, "r") as f:
                # f = g["US"]["US_DATASET0000"]
                self.angles = np.array(f['angles'])  # (nangles,)
                self.bimg = np.array(f["beamformed_data"], dtype="float32")
                # self.fs = np.array(f["beamformed_data_sampling_frequency"], dtype="float32").item()
                # self.time_zero = np.array(f["beamformed_data_t0"]).squeeze()  # (nagles,)

                self.idata = np.array(f["channel_data"], dtype="float32")  # (nangles, nXDCs, nsamples)
                self.qdata = None
                self.fs = np.array(f["channel_data_sampling_frequency"], dtype="float32").item()
                self.time_zero = np.array(f["channel_data_t0"]).squeeze()  # (nagles,)
                self.fc = np.array(f['modulation_frequency']).item()
                self.c = np.array(f["sound_speed"]).item()
                # self.nXDCs = np.array(f["element_count"]).item()
                xpos = (np.linspace(-31.36/2, 31.36/2, 128)/1000)[:, None]
                self.ele_pos = np.concatenate([xpos, np.zeros_like(xpos), np.zeros_like(xpos)], axis=1)     # (nxDCs, 3)
                # self.pixel_positions = np.array(f['pixel_positions'])  # (3, H, W)
                # self.nangles = np.array(f['transmit_count']).item()
                # # grid info
                # self.grid_xlims = [self.ele_pos[0, 0], self.ele_pos[-1, 0]]
                # self.grid_zlims = None
                modulation = "rf"
        elif self.code == 'INS':
            with h5py.File(self.fname, "r") as f:
                # f = g["US"]["US_DATASET0000"]
                self.bimg = np.array(f["beamformed_data"], dtype="float32")
                self.idata = np.array(f["channel_data"], dtype="float32")   # (nangles, nXDCs, nsamples)
                self.qdata = None
                self.nXDCs = np.array(f["element_count"]).item()
                self.ele_pos = np.array(f["element_positions"]).T   # (nxDCs, 3)
                self.fc = np.array(f['modulation_frequency']).item()
                self.pixel_positions = np.array(f['pixel_positions']) #(3, H, W)
                self.fs = np.array(f["sampling_frequency"]).item()
                self.c = np.array(f["sound_speed"]).item()
                # self.c = 1473.0
                self.time_zero = np.array(f["start_time"]).squeeze()    # (nagles,)
                self.nangles = np.array(f['transmit_count']).item()
                self.angles = np.array(f['transmit_direction'])[0]      # (nangles,)
                # grid info
                self.grid_xlims = [self.ele_pos[0, 0], self.ele_pos[-1, 0]]
                self.grid_zlims = None
                modulation = "rf"
        self.fdemod = self.fc if modulation == "iq" else 0

        # If data is RF, use the Hilbert transform to get the imaginary component.
        if modulation == "rf":
            iqdata = hilbert(self.idata, axis=-1)
            self.qdata = np.imag(iqdata)

        # Make sure that time_zero is an array of size [nangles]
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero
        # Validate that all information is properly included
        super().validate()


class LoadDataPICMUS(PlaneWaveData):
    def __init__(self, dataset_path, task, filename):
        # Load dataset
        self.dir = os.path.join(dataset_path, task)
        self.filename = filename
        self.fname = os.path.join(self.dir, self.filename)
        with h5py.File(self.fname, "r") as g:
            f = g["US"]["US_DATASET0000"]
            self.idata = np.array(f["data"]["real"], dtype="float32")
            self.qdata = np.array(f["data"]["imag"], dtype="float32")
            self.angles = np.array(f["angles"])
            self.fc = 5208000.0  # np.array(f["modulation_frequency"]).item()
            self.fs = np.array(f["sampling_frequency"]).item()
            self.c = np.array(f["sound_speed"]).item()
            self.time_zero = np.array(f["initial_time"])
            self.ele_pos = np.array(f["probe_geometry"]).T
            # grid info
            self.grid_xlims = [self.ele_pos[0, 0], self.ele_pos[-1, 0]]
            self.grid_zlims = None

            modulation = "rf"
        self.fdemod = self.fc if modulation == "iq" else 0
        # If data is RF, use the Hilbert transform to get the imag. component.
        if modulation == "rf":
            iqdata = hilbert(self.idata, axis=-1)
            self.qdata = np.imag(iqdata)
        # Make sure that time_zero is an array of size [nangles]
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero
        # Validate that all information is properly included
        super().validate()

class LoadData(PlaneWaveData):
    def __init__(self, orig, acq, target, modulation, param):
        """ Load dataset as a PlaneWaveData object. """
        # Make sure the selected dataset is valid
        assert any([orig == o for o in ["cubdl", "fieldII"]])
        assert any([acq == a for a in ["simulation", "experiments"]])
        # assert any([target == t for t in ["contrast_speckle", "resolution_distorsion"]])
        assert any([modulation == d for d in ["rf", "iq"]])

        datasets_path = GV.DATASETS_PATH

        # Load dataset
        if orig == "cubdl":
            self.dir = os.path.join(datasets_path, orig, acq, target)
            self.filename = "%s_%s_dataset_%s.hdf5" % (target, acq[:4], modulation)
            fname = os.path.join(self.dir, self.filename)
            with h5py.File(fname, "r") as g:
                f = g["US"]["US_DATASET0000"]
                self.idata = np.array(f["data"]["real"], dtype="float32")
                self.qdata = np.array(f["data"]["imag"], dtype="float32")
                self.angles = np.array(f["angles"])
                self.fc = 5208000.0  # np.array(f["modulation_frequency"]).item()
                self.fs = np.array(f["sampling_frequency"]).item()
                self.c = np.array(f["sound_speed"]).item()
                self.time_zero = np.array(f["initial_time"])
                self.ele_pos = np.array(f["probe_geometry"]).T
                # grid info
                self.grid_xlims = [self.ele_pos[0, 0], self.ele_pos[-1, 0]]
                self.grid_zlims = [5e-3, 55e-3]
        elif orig == "fieldII":
            self.dir = os.path.join(datasets_path, orig, acq, target)
            self.filename = "%s_%s_dataset_%s_%s.h5" % (target, acq[:4], modulation, param)
            fname = os.path.join(self.dir, self.filename)
            with h5py.File(fname, "r") as f:
                idata = np.array(f["data"]["real"], dtype="float32")
                self.idata = idata / np.amax(idata)
                self.qdata = 0  # This will be filled below
                if modulation == 'iq':
                    qdata = np.array(f["data"]["imag"], dtype="float32")
                    self.qdata = qdata / np.amax(qdata)
                self.angles = np.squeeze(np.array(f['angles']) * np.pi / 180)
                self.fc = np.array(f['fc']).item()
                self.fs = np.array(f['fs']).item()
                self.c = np.array(f['c']).item()
                self.time_zero = np.squeeze(np.array(f['time_zero']))
                xs = np.squeeze(np.array(f['ele_pos']))
                self.ele_pos = np.array([xs, np.zeros_like(xs), np.zeros_like(xs)]).T
                # load phantom info
                self.phantom_amp = np.array(f['phantom']['amplitudes'])
                self.phantom_pos = np.array(f['phantom']['positions'])
                # grid info
                self.grid_xlims = [np.min(self.phantom_pos[0]), np.max(self.phantom_pos[0])]
                self.grid_zlims = [np.min(self.phantom_pos[2]), np.max(self.phantom_pos[2])]
        self.fdemod = self.fc if modulation == "iq" else 0
        # If data is RF, use the Hilbert transform to get the imag. component.
        if modulation == "rf":
            iqdata = hilbert(self.idata, axis=-1)
            self.qdata = np.imag(iqdata)
        # Make sure that time_zero is an array of size [nangles]
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero
        # Validate that all information is properly included
        super().validate()


class LoadData_nair2020(PlaneWaveData):
    def __init__(self, h5_dir, simu_name):
        # raw_dir = 'D:\Itamar\\datasets\\fieldII\\simulation\\nair2020\\raw'
        # raw_dir = 'D:\\Itamar\\datasets\\fieldII\\simulation\\nair2020\\raw12500_0.5attenuation'
        simu_number = int(simu_name[4:])
        lim_inf = 1000*((simu_number-1)//1000) + 1
        lim_sup = lim_inf + 999
        h5_name = 'simus_%.5d-%.5d.h5' % (lim_inf, lim_sup)
        h5filename = os.path.join(h5_dir, h5_name)
        # print(h5filename)
        with h5py.File(h5filename, "r") as g:
        # g = h5py.File(filename, "r")
            f = g[simu_name]
            self.idata = np.expand_dims(np.array(f["signal"], dtype="float32"), 0)
            self.qdata = np.imag(hilbert(self.idata, axis=-1))
            self.angles = np.array([0])
            self.fc = np.array(f['fc']).item()
            self.fs = np.array(f['fs']).item()
            self.c = np.array(f['c']).item()
            self.time_zero = np.array([np.array(f['time_zero']).item()])
            self.fdemod = 0
            xs = np.squeeze(np.array(f['ele_pos']))
            self.grid_xlims = [xs[0], xs[-1]]
            self.grid_zlims = [30*1e-3, 80*1e-3]
            self.ele_pos = np.array([xs, np.zeros_like(xs), np.zeros_like(xs)]).T
            self.pos_lat = np.array(f['lat_pos']).item()
            self.pos_ax = np.array(f['ax_pos']).item()
            self.radius = np.array(f['r']).item()
        super().validate()

class LoadData_goudarzi2020(PlaneWaveData):
    def __init__(self, h5_dir, h5_name):
        # super().__init__(self)
        h5filename = os.path.join(h5_dir, h5_name)
        with h5py.File(h5filename, "r") as f:
            # self.angles = np.linspace(-0.2793, 0.2793, 75)
            self.angles = np.array([0])
            self.fc = np.array(f['fc']).item()
            self.fs = np.array(f['fs']).item()
            self.c = np.array(f['c']).item()
            self.ele_pos = np.array(f['ele_pos']).T       # 128 x 3
            self.grid_xlims = np.array(f['grid_x'])     # 1 x 387
            self.grid_zlims = np.array(f['grid_z'])     # 1 x 689
            self.data_IQ = np.array(f["signal"])
            print(self.data_IQ.shape)
            # self.idata = self.data_IQ[len(self.data_IQ)//2][None]   # 1 x 128 x 3328
            # self.qdata = np.imag(hilbert(self.idata, axis=-1))      # 1 x 128 x 3328
            self.idata = self.data_IQ[None]   # 1 x 128 x 3328
            self.qdata = np.imag(hilbert(self.idata, axis=-1))      # 1 x 128 x 3328
            self.time_zero = np.array([np.array(f['time_zero']).item()])
            self.fdemod = 0
            self.bdata = np.array(f['bdataIQ'])         # 2 x 235
            # 683
        self.validate()
    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles
        print("Dataset successfully loaded")


class LoadData_goudarzi2020_MVB(PlaneWaveData):
    def __init__(self, h5_dir, h5_name):
        # super().__init__(self)
        h5filename = os.path.join(h5_dir, h5_name)
        with h5py.File(h5filename, "r") as f:
            # self.angles = np.linspace(-0.2793, 0.2793, 75)
            self.angles = np.array([0])
            self.fc = np.array(f['fc']).item()
            self.fs = np.array(f['fs']).item()
            self.c = np.array(f['c']).item()
            self.ele_pos = np.zeros((128, 3))
            self.ele_pos[:, 0] = np.array(f['ele_pos']).flatten()       #
            self.grid_xlims = np.array(f['grid_x'])     #
            self.grid_zlims = np.array(f['grid_z'])     #
            self.data_IQ = np.array(f["channel_data"])
            print(self.data_IQ.shape)
            # self.idata = self.data_IQ[len(self.data_IQ)//2][None]   #
            # self.qdata = np.imag(hilbert(self.idata, axis=-1))      #
            self.idata = self.data_IQ[None]   # 1 x 128 x 3328
            self.qdata = np.imag(hilbert(self.idata, axis=-1))      #
            self.time_zero = np.array([np.array(f['time_zero']).item()])
            self.fdemod = 0
            self.bdata_mvbI = np.array(f['bdata_mvbI'])         #
            self.bdata_mvbQ = np.array(f['bdata_mvbQ'])         #
            # 683
        self.validate()
    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles
        print("Dataset successfully loaded")


class LoadData_phantomLIM_ATSmodel539(PlaneWaveData):
    def __init__(self, h5_dir, h5_name):
        # super().__init__(self)
        h5filename = os.path.join(h5_dir, h5_name)
        with h5py.File(h5filename, "r") as f:
            self.angles = np.array([0])
            # self.angles = np.array(f['angles']).squeeze()
            self.fc = np.array(f['fc']).item()
            # self.fc = np.array(5*1e6).item()
            self.fs = np.array(f['fs']).item()
            # self.fs = np.array(29.6*1e6).item()
            self.c = np.array(f['c']).item()
            self.x = np.array(f['x'])      # x 1x128 in MATLAB
            self.z = np.array(f['z'])      # z 1x2048 in MATLAB
            xs = np.squeeze(self.x)
            self.ele_pos = np.array([xs, np.zeros_like(xs), np.zeros_like(xs)]).T
            self.data_IQ = np.array(f["rf"], dtype="float32")[3]    # 2048x128 in MATLAB
            # print(f["rf"][3])
            # print(f["rf"][3].shape)
            #
            # self.data_IQ = f["rf"][3]    # 2048x128 in MATLAB
            # self.idata = np.expand_dims(np.array(f["signal"], dtype="float32"), 0)
            # self.qdata = np.imag(hilbert(self.idata, axis=-1))
            # print(self.data_IQ.shape)
            # self.idata = self.data_IQ[len(self.data_IQ)//2][None]   # 1 x 128 x 3328
            # self.qdata = np.imag(hilbert(self.idata, axis=-1))      # 1 x 128 x 3328
            self.idata = self.data_IQ[None]  # 1 x 128 x 3328
            # self.idata = self.data_IQ   # 1 x 128 x 3328
            self.qdata = np.imag(hilbert(self.idata, axis=-1))      # 1 x 128 x 3328
            # self.time_zero = np.array([np.array(1e-6).item()])
            self.time_zero = np.array([np.array(f['time_zero'][4]).item()])
            # self.time_zero = np.array(f['time_zero']).squeeze()
            self.fdemod = 0
            r = 8 * 1e-3
            xctr = 0.0
            zctr = 40 * 1e-3
            self.pos_lat = xctr
            self.pos_ax = zctr
            self.radius = r
            # self.grid_zlims = [0.002, 0.052]
            # self.grid_zlims = [0.002, 0.055]
            self.phantom_zlims = [0.001, 0.08]
            self.phantom_xlims = [-0.019, 0.019]
        self.validate()
    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles
        print("Dataset successfully loaded")


if __name__ == '__main__':
    # DATASETS_PATH = '/home/lim/Documents/Itamar/datasets'
    # DATASETS_PATH = 'D:/Itamar/datasets'

    print("*** Loading dataset ***")
    # P = LoadData(orig='cubdl', acq='simulation',
    #              target='contrast_speckle', modulation='rf', param='')
    # P = LoadData(orig='fieldII', acq='simulation',
    #              target='contrast_speckle_01', modulation='rf', param='5.2MHz')

    P = LoadDataPICMUS(dataset_path='/home/lim/itamar/datasets/',
                       orig='cubdl',
                       acq='simulation',
                       target='contrast_speckle',
                       modulation='rf')
    print(P.fs)
