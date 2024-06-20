import numpy as np
from glob import glob
from skimage.io import imsave, imread
from datetime import datetime

class Reader:
    def __init__(self, freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, power, bits, globstrs_1bit, folder, roi_dim):
        self.freq = freq
        self.frames = frames
        self.gate_num = gate_num
        self.gate_integ = gate_integ
        self.gate_width = gate_width
        self.gate_step = gate_step
        self.gate_offset = gate_offset
        self.power = power
        self.bits = bits
        self.globstrs_1bit = globstrs_1bit
        self.folder = folder
        self.roi_dim = roi_dim
        self.filename = self.name()

    def name(self):
        date = datetime.now().strftime('%y%m%d')
        filename = f'{date}_SPAD-QD-{self.freq}MHz-{self.frames}f-{self.gate_num}g-{int(self.gate_integ * 1e3)}us-{self.gate_width}ns-{int(self.gate_step * 1e3)}ps-{int(self.gate_offset * 1e3)}ps-{self.power}uW.tif'
        return filename

    def read_bin(self, globstr, nframes=1000):
        files = glob(globstr)
        stacks = []
        for file in files:
            byte = np.fromfile(file, dtype='uint8')
            bits = np.unpackbits(byte)
            bits = np.array(np.split(bits, nframes))
            bits = bits.reshape((nframes, 512, 512)).swapaxes(1, 2)
            bits = np.flip(bits, axis=1)
            stacks.append(bits)
        stack = np.concatenate(stacks, axis=0)
        return stack

    def stack_1bit(self):
        for n, globstr in enumerate(self.globstrs_1bit):
            stack = self.read_bin(globstr, nframes=1000)
            imsave(f'{self.filename}_stack{n}.tif', stack)

    def stack(self):
        files = sorted(glob(f'{self.folder}/*.png'))
        stack = np.array([imread(f) for f in files])
        imsave(f'{self.filename}.tif', stack[:, :self.roi_dim, :self.roi_dim])

    def process(self):
        if self.bits == 1:
            self.stack_1bit()
        else:
            self.stack()

# # acq params for naming
# freq = 10  # frequency in MHz
# frames = 3  # number of frames
# gate_num = 1000  # number of gates per frame
# gate_integ = 10  # integration time in ms
# gate_width = 5  # gate width in ns
# gate_step = 0.018  # gate step size in ns
# gate_offset = 0.018  # gate offset in ns
# power = 150  # pulsed laser power in uW

# # 1bit params
# bits = 1  # bit depth of data acquisition
# globstrs_1bit = ['RAW0000*.bin*']  # filename format for glob to read when stacking 1-bit images

# # non-1bit params
# folder = 'acq00001'  # folder name with images, no slash at end
# roi_dim = 256  # code saves only a square with size roi_dim from the top left of acquisitions

# # flim_reader = Reader(freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, power, bits, globstrs_1bit, folder, roi_dim)
# # flim_reader.process()
