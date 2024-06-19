from datetime import datetime

'''
Function for auto naming acquisitons. Be careful with units.
'''
freq = 10 # frequency in MHz
frames = 3 # number of frames
gate_num = 1000 # number of gates per frame
gate_integ = 10 # integration time in ms
gate_width = 5 # gate width in ns
gate_step = 0.018 # gate step size in ns
gate_offset = 0.018 # gate offset in ns
power = 150 # pulsed laser power in uW

def name(freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, power):
    date =  datetime.now().strftime('%y%m%d')
    filename = f'{date}_SPAD-QD-{freq}MHz-{frames}f-{gate_num}g-{int(gate_integ*1e3)}us-{gate_width}ns-{int(gate_step*1e3)}ps-{int(gate_offset*1e3)}ps-{power}uW.tif'
    return filename

filename = name(freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, power)
print(filename)