import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imsave, imread
from datetime import datetime

class IntensityReaderSparse:
    def __init__(self, config):
        self.path = config['path']
        self.savepath = config['savepath']
        self.prefix = config['prefix']
        self.roi_dim = config['roi_dim']
        self.filename = self.name()

    def name(self):
        filename = self.savepath + self.prefix
        return filename

    def read_1bit_sparse(self,globstr='RAW*',dummy=262144):
        summed = np.zeros((512*512,))
        files = sorted(glob(self.path+globstr))
        for f in files:
            print('Reading: ' + f)
            bytes = np.fromfile(f,dtype='uint32')
            idx = np.where(bytes == dummy)[0]
            for n in range(len(idx)):
                if n > 0:
                    this_frame = bytes[idx[n-1]:idx[n]]
                else:
                    this_frame = bytes[:idx[n]]
                this_frame = this_frame[1:]
                summed[this_frame] += 1
        summed = np.reshape(summed,(512,512))
        summed = np.rot90(summed)
        return summed

    def read_1bit_sparse_roi(self,xmin,xmax,ymin,ymax,
                             globstr='RAW*',dummy=262144):
        stack = []
        files = sorted(glob(self.path+globstr))
        for f in files:
            bytes = np.fromfile(f,dtype='uint32')
            idx = np.where(bytes == dummy)[0]
            for n in range(len(idx)):
                print(f'Reading frame {n}')
                frame = np.zeros((512*512),dtype='uint8')
                if n > 0:
                    this_frame = bytes[idx[n-1]:idx[n]]
                else:
                    this_frame = bytes[:idx[n]]
                this_frame = this_frame[1:]
                frame[this_frame] += 1
                frame = np.reshape(frame,(512,512))
                frame = np.rot90(frame)
                stack.append(frame[xmin:xmax,ymin:ymax])
                del frame
        stack = np.array(stack)
        return stack

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
        stack = stack[:,self.roi_dim:,self.roi_dim:]
        imsave(f'{self.filename}.tif', stack)
        return stack

    def stack(self):
        files = sorted(glob(f'{self.path}/*.png'))
        stack = np.array([imread(f) for f in files])
        imsave(f'{self.filename}.tif', stack)


class GatedReader:
    def __init__(self, config):
        self.freq = config['freq']
        self.frames = config['frames']
        self.gate_num = config['numsteps']
        self.gate_integ = config['integ']
        self.gate_width = config['width']
        self.gate_step = config['step']
        self.gate_offset = config['offset']
        self.power = config['power']
        self.bits = config['bits']
        self.folder = config['folder']
        self.filename = self.name()

    def name(self):
        date = datetime.now().strftime('%y%m%d')
        self.filename = f'{date}_SPAD-{self.freq}MHz-{self.frames}f-{self.gate_num}g-{int(self.gate_integ)}us-{self.gate_width}ns-{int(self.gate_step)}ps-{int(self.gate_offset)}ps-{self.power}uW'
        return self.filename

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

    def stack(self):
        files = sorted(glob(f'{self.folder}/*.png'))
        stack = np.array([imread(f) for f in files])
        return stack

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
