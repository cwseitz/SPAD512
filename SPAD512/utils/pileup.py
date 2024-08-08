import numpy as np
import random

lifetimes = [10]
zeta = 0.05       
offset = 0.018     
step = 1             
width = 5        
freq = 10      
max_pileups = 3 
numgates = 10000           

for i in range(numgates):
    lam = 1 / lifetimes[0]
    prob = []
    for j in range(max_pileups):
        prob_value = zeta * (
            np.exp(-lam * (offset + step + j * (1e3 / freq))) -
            np.exp(-lam * (offset + step + width + j * (1e3 / freq)))
        )
        prob.append(prob_value)

    prob.append(1 - sum(prob))

    det_pulse = random.choices(range(len(prob)), weights=prob)[0]
    
    print(f"Gate {i+1}: Probabilities = {prob}, Selected index = {det_pulse}")
