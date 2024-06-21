import numpy as np
from glob import glob
from skimage.io import imsave, imread
from datetime import datetime

class IntensityReader:
    def __init__(self, config):
        self.path = config['path']
        self.savepath = config['savepath']
        self.roi_dim = config['roi_dim']
        self.prefix = config['prefix']
        self.filename = self.name()

    def name(self):
        filename = self.savepath + self.prefix
        return filename

    def read_bin(self,globstr,nframes=1000):
        """nframes per .bin file (typically 1000/file)"""
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

    def stack_1bit(self,globstrs='RAW*',nframes=1000):
        binfiles = glob(self.path+globstrs)
        out = []
        for n, binfile in enumerate(binfiles):
            this_stack = self.read_bin(binfile, nframes=nframes)
            out.append(this_stack)
        stack = np.concatenate(np.array(out),axis=0)
        imsave(f'{self.filename}.tif', stack)

    def stack(self):
        files = sorted(glob(f'{self.path}/*.png'))
        stack = np.array([imread(f) for f in files])
        imsave(f'{self.filename}.tif', stack[:, :self.roi_dim, :self.roi_dim])


class GatedReader:
    def __init__(self, config):
        self.freq = config['freq']
        self.frames = config['frames']
        self.gate_num = config['gate_num']
        self.gate_integ = config['gate_integ']
        self.gate_width = config['gate_width']
        self.gate_step = config['gate_step']
        self.gate_offset = config['gate_offset']
        self.power = config['power']
        self.bits = config['bits']
        self.globstrs_1bit = config['globstrs_1bit']
        self.folder = config['folder']
        self.roi_dim = config['roi_dim']
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
        """shouldn't be needed for gated acquisitions"""
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
    
    def parse(filename):
        # split filename into individual values
        base = filename.split('/')[-1]
        base = base.split('.')[0]
        parts = base.split('-')
        
        # extract parameter values
        freq = int(parts[2].replace('MHz', ''))
        frames = int(parts[3].replace('f', ''))
        gate_num = int(parts[4].replace('g', ''))
        gate_integ = int(parts[5].replace('us', ''))
        gate_width = int(parts[6].replace('ns', ''))
        gate_step = float(parts[7].replace('ps', '')) / 1000  # Convert from ps to ns
        gate_offset = float(parts[8].replace('ps', '')) / 1000  # Convert from ps to ns

        return freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset
