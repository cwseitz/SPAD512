from SPAD512S import SPAD512S
import time
import numpy as np

def analyze_data(img):
    mean_val = np.mean(img)
    
    if mean_val < 20:
        return +10
    else:
        return -10


def run_adaptive_acquisition(
    port=9999,          # tcp port, need to change depending on where the software is listening
    initial_gate_offset=0,  
    gate_steps=5,          
    gate_step_size=50,    
    gate_width=1,          
    iterations=1,          
    bit_depth=8,           
    measure_time=5,        
    overlap=0,            
    pileup=1,           
    max_cycles=10       
):
    spad = SPAD512S(port)
    
    stop = bytes('STOP', 'utf-8') # make sure the camera is in a known state
    spad.t.send(stop)
    _ = spad.t.recv(8192).decode('utf-8')  # read acknowledgment 
    
    print("Connected to SPAD512S and stopped any prior acquisitions.")
    
    gate_offset = initial_gate_offset
    
    for cycle_idx in range(max_cycles):
        print(f'adaptive cycle {cycle_idx+1}/{max_cycles}')
        print(f"current offset: {gate_offset} ps")
        
        img = spad.get_gated_intensity(
            bitDepth=bit_depth,
            intTime=measure_time,
            iterations=iterations,
            gate_steps=gate_steps,
            gate_step_size=gate_step_size,
            gate_step_arbitrary=0,  
            gate_width=gate_width,
            gate_offset=gate_offset,
            gate_direction=0,    
            gate_trig=0,          
            overlap=overlap,
            stream=1,             
            pileup=pileup,
            im_width=512           
        )
        
        offset_adjustment = analyze_data(img)
        
        gate_offset += offset_adjustment
        
        if gate_offset < 0:
            gate_offset = 0
        if gate_offset > 5000:
            gate_offset = 5000
        
        # put spad.set_arbitrary_steps([...]) here 
        
        # maybe add an auto-breaking condition if new data points are unhelpful


if __name__ == "__main__":
    run_adaptive_acquisition(
        port=9999,             
        initial_gate_offset=0,  
        gate_steps=5,         
        gate_step_size=50,      
        gate_width=1,         
        iterations=1,          
        bit_depth=8,            
        measure_time=5,        
        overlap=0,            
        pileup=1,              
        max_cycles=10           
    )
