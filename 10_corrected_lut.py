import numpy as np
import matplotlib.pyplot as plt

# Typical gamma value used for sRGB
gamma = 2.2

# 0–255 input values
k = np.arange(256)

# Apply gamma-correction formula:
lut = np.round(255 * (k / 255) ** (1 / gamma))

# Plot the curve
plt.figure(figsize=(6,4))
plt.plot(k, lut, 'k.')      
plt.plot(k, k, 'k--')        
plt.xlim(0,255)
plt.ylim(0,255)
plt.xlabel('Input value (k)')
plt.ylabel('Gamma-corrected value')
plt.title('Gamma Correction Lookup Table')
plt.grid(True)
plt.show()
