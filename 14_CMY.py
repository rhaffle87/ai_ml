import numpy as np
import matplotlib.pyplot as plt

# Rentang panjang gelombang (nm)
wavelengths = np.linspace(400, 700, 300)

# Buat kurva sensitivitas sederhana untuk RGB
R = np.exp(-0.5*((wavelengths-600)/30)**2)   # merah
G = np.exp(-0.5*((wavelengths-550)/30)**2)   # hijau
B = np.exp(-0.5*((wavelengths-450)/30)**2)   # biru

# Kurva CMY sebagai komplemen RGB
C = G + B     # Cyan = Green + Blue
M = R + B     # Magenta = Red + Blue
Y = R + G     # Yellow = Red + Green

# Normalisasi agar tetap dalam rentang [0,1]
C /= C.max(); M /= M.max(); Y /= Y.max()

plt.figure(figsize=(8,5))

# Plot RGB
plt.plot(wavelengths, R, 'r--', label='R (RGB)')
plt.plot(wavelengths, G, 'g--', label='G (RGB)')
plt.plot(wavelengths, B, 'b--', label='B (RGB)')

# Plot CMY
plt.plot(wavelengths, C, 'c', label='C (CMY)')
plt.plot(wavelengths, M, 'm', label='M (CMY)')
plt.plot(wavelengths, Y, 'y', label='Y (CMY)')

plt.title("Kurva Sensitivitas Spektral CMY vs RGB")
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Sensitivitas (relatif)")
plt.legend()
plt.grid(True)
plt.show()
