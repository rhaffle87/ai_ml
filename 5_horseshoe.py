import numpy as np
import matplotlib.pyplot as plt
import csv

# Membaca data dari file CSV CIE 1931 (x_bar, y_bar, z_bar)
wavelengths = []
x_bar = []
y_bar = []
z_bar = []
with open('CIE_xyz_1931_2deg.csv', 'r') as f:
    reader = csv.reader(f)
    # jika CSV punya header, skip baris header
    next(reader)
    for row in reader:
        wl = float(row[0])   # panjang gelombang (nm)
        x = float(row[1])
        y = float(row[2])
        z = float(row[3])
        # Tambahkan hanya di rentang terlihat (misal 380–780 nm)
        if 380.0 <= wl <= 780.0:
            wavelengths.append(wl)
            x_bar.append(x)
            y_bar.append(y)
            z_bar.append(z)

# Hitung locus kromatisitas
locus = []
for X, Y, Z in zip(x_bar, y_bar, z_bar):
    denom = X + Y + Z
    if denom > 0:
        x = X / denom
        y = Y / denom
        locus.append((x, y))

locus = np.array(locus)

# Plot
plt.figure(figsize=(6,6))
plt.plot(locus[:,0], locus[:,1], '-', color='blue')
plt.title('Spectrum Locus CIE 1931 (2° observer)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()