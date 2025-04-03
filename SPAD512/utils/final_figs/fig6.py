import matplotlib.pyplot as plt
import numpy as np

v_usb3 = 2.4e9  
t_r = 10e-6  

parameter_sets = [
    {"label": "4-bit ", "X": 512, "G": 100, "N": 4, "F": 200, "TI": 1e-4}, 
    {"label": "6-bit ", "X": 512, "G": 100, "N": 6, "F": 40, "TI": 5e-4}, 
    {"label": "8-bit ", "X": 512, "G": 100, "N": 8, "F": 10, "TI": 20e-4},   
    {"label": "10-bit ", "X": 512, "G": 100, "N": 10, "F": 2, "TI": 100e-4},  
    {"label": "12-bit ", "X": 512, "G": 100, "N": 12, "F": 1, "TI": 200e-4},   # temporal

    {"label": "4-bit", "X": 512, "G": 100, "N": 4, "F": 1, "TI": 200e-4}, 
    {"label": "6-bit", "X": 512, "G": 100, "N": 6, "F": 1, "TI": 200e-4}, 
    {"label": "8-bit", "X": 512, "G": 100, "N": 8, "F": 1, "TI": 200e-4},   
    {"label": "10-bit", "X": 512, "G": 100, "N": 10, "F": 1, "TI": 200e-4},  
    {"label": "12-bit", "X": 512, "G": 100, "N": 12, "F": 1, "TI": 200e-4},   # temporal
    

    # {"label": "12-bit (T.B.)", "X": 512, "G": 100, "N": 12, "F": 1, "TI": 0.02},    #
    # {"label": "4-bit (S.B.)", "X": 512, "G": 100, "N": 4, "F": 1, "TI": 2.0},      # 2.0s
    # {"label": "2-bit (T.B.)", "X": 512, "G": 100, "N": 2, "F": 400, "TI": 0.005},  # 2.0s
    # {"label": "3-bit (T.B.)", "X": 512, "G": 100, "N": 3, "F": 300, "TI": 0.00667},# 2.0s
    # {"label": "6-bit (T.B.)", "X": 512, "G": 100, "N": 6, "F": 100, "TI": 0.02},   # 2.0s
    # {"label": "8-bit (T.B.)", "X": 512, "G": 100, "N": 8, "F": 50, "TI": 0.04},    # 2.0s
    # {"label": "10-bit (T.B.)", "X": 512, "G": 100, "N": 10, "F": 20, "TI": 0.1},   # 2.0s
    # {"label": "2-bit (S.B.)", "X": 512, "G": 100, "N": 2, "F": 1, "TI": 2.0},      # 2.0s
    # {"label": "3-bit (S.B.)", "X": 512, "G": 100, "N": 3, "F": 1, "TI": 2.0},      # 2.0s
    # {"label": "6-bit (S.B.)", "X": 512, "G": 100, "N": 6, "F": 1, "TI": 2.0},      # 2.0s
    # {"label": "8-bit (S.B.)", "X": 512, "G": 100, "N": 8, "F": 1, "TI": 2.0},      # 2.0s
    # {"label": "10-bit (S.B.)", "X": 512, "G": 100, "N": 10, "F": 1, "TI": 2.0},    # 2.0s
]

all_results = []
for params in parameter_sets:
    X, G, N, F, TI = params["X"], params["G"], params["N"], params["F"], params["TI"]
    T_dead = F * ((X**2 * G * N) / v_usb3 + G * (2**N - 1) * t_r)
    print(F*((2**N - 1) * t_r))
    T_active = F * G * TI
    all_results.append((params["label"], T_dead, T_active))

# all_results.sort(key=lambda x: x[1])

labels = [r[0] for r in all_results]
readout_times = np.array([r[1] for r in all_results])
active_times = np.array([r[2] for r in all_results])

plt.figure(figsize=(10, 6))
bar_width = 0.5
plt.bar(labels, active_times, bar_width, label="Active Time", color="darkgreen")
plt.bar(labels, readout_times, bar_width, bottom=active_times, label="Readout Time", color="red", alpha=0.5)
plt.ylabel("Time (s)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()
