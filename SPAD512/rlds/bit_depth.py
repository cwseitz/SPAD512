import numpy as np
import matplotlib.pyplot as plt

'''Bi-exponential Rapid Lifetime Determination formula'''
def bi_rld(data, g, s):
    D0, D1, D2, D3 = data

    R = D1*D1 - D2*D0
    P = D3*D0 - D2*D1
    Q = D2*D2 - D3*D1
    disc = P**2 - 4*R*Q
    y = (-P + np.sqrt(disc))/(2*R)
    x = (-P - np.sqrt(disc))/(2*R)
    S = s * ((x**2)*D0 - (2*x*D1) + D2)
    T = (1-((x*D1 - D2)/(x*D0 - D1))) ** (g/s)

    tau1 = -s/np.log(y)
    tau2 = -s/np.log(x)

    A1 = (-(x*D0 - D1)**2) * np.log(y) / (S * T) 
    A2 = (-R * np.log(x)) / (S * ((x**(g/s)) - 1))

    return (A1, tau1, A2, tau2)

'''Data "generation" by directly calculating probabilities'''
def get_prob(reg, params, zeta=0.01):
    start, stop = reg
    A1, tau1, A2, tau2 = params

    A = A1/(A1+A2)
    lam1 = 1/tau1
    lam2 = 1/tau2

    prob = zeta * (-A*np.exp(-stop*lam1) -(1-A)*np.exp(-stop*lam2)
                   + A*np.exp(-start*lam1) + (1-A)*np.exp(-start*lam2))
    return prob

def gen_step(prob,integ=10,bits=8,freq=10,dist=True):
    tot_gates = integ*freq*1e3
    bin_gates = int(tot_gates/(2**bits))

    temp = 1-((1-prob)**bin_gates)
    bin_prob = 1-(temp**(1/bin_gates))
    
    if dist:
        return np.random.binomial(2**bits, bin_prob)
    else:
        return (2**bits)*bin_prob


'''Set Parameters'''
params = [1, 10, 2, 20] # [amp1, tau1, amp2, tau2]
regs = [[3, 8], [5, 10], [7, 12], [9, 14]] # gate regions manually specified for simplicity; gate width is 5 here, and gate step is 2
g = regs[0][1] - regs[0][0] # gate width is length of each gate region
s = regs[1][0] - regs[0][0] # gate step is the difference between gate opening times
integ = 10 # integration time in ms, for bit-depth simulating
bits = 8 # bit-depth, for bit-depth simulating
freq = 10 # frequency in MHz, for bit-depth simulating
bin_gates = int(1e3*freq*(integ/(2**bits))) # number of gate repetitions for a single binary image within a single gate step

'''Generate Data'''
data = []
for reg in regs:
    prob = get_prob(reg, params)
    data.append(gen_step(prob))

'''Correct via derived formula'''
n_data = []
for dt in data:
    temp = 1-dt/(2**bits)
    n_data.append(1-(temp**(1/bin_gates)))
A1, tau1, A2, tau2 = np.round(bi_rld(n_data, g, s), 5)
print(f'Recovered {tau1, tau2}') 
print(f'Lifetime error: {params[1] - tau1, params[3] - tau2}\n')

# '''OLD CODE FROM EMAIL'''
# '''Test bi-exponential RLD'''
# probs = []
# for reg in regs:
#     probs.append(get_prob(reg, params)) # can also add a scaling factor here, doesn't change the math
# A1, tau1, A2, tau2 = np.round(bi_rld(probs, g, s), 5)
# print(f'Results of bi-exponential RLD with no bit-depth simulating: {tau1, tau2}') 
# print(f'Lifetime error: {params[1] - tau1, params[3] - tau2}\n')

# '''Test same method but with bit-depth saturated data'''
# bin_gates = int(1e3*freq*(integ/(2**bits))) # number of gate repetitions for a single binary image within a single gate step
# bit_probs = []
# for prob in probs:
#     temp = 1-((1-prob)**bin_gates)
#     bit_probs.append((2**bits) * temp) # (1-prob)^bin_gates gives probability of no counts for the whole binary image, so take complement again
# A1, tau1, A2, tau2 = np.round(bi_rld(bit_probs, g, s), 5)
# print(f'With bit-depth simulating: {tau1, tau2}') 
# print(f'Lifetime error: {params[1] - tau1, params[3] - tau2}\n')

# '''Recover initial lifetimes by reversing complementary data generation'''
# re_probs = []
# for bit_prob in bit_probs:
#     temp = 1-bit_prob/(2**bits)
#     re_probs.append(1-(temp**(1/bin_gates)))
# A1, tau1, A2, tau2 = np.round(bi_rld(re_probs, g, s), 5)
# print(f'Recovered {tau1, tau2}') 
# print(f'Lifetime error: {params[1] - tau1, params[3] - tau2}\n')