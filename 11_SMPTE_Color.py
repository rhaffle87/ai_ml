import numpy as np
import colour
import matplotlib.pyplot as plt
from colour.plotting import plot_chromaticity_diagram_CIE1931

# Ambil data SMPTE 240M
cs = colour.RGB_COLOURSPACES['SMPTE 240M']

# Ambil primaries (x, y)
primaries = cs.primaries
red, green, blue = primaries

# Ambil white point (D65)
white = cs.whitepoint

# Plot background horseshoe CIE 1931
plot_chromaticity_diagram_CIE1931(standalone=False)

# Plot segitiga primaries
triangle = np.vstack([red, green, blue, red])  # tutup segitiga
plt.plot(triangle[:,0], triangle[:,1], 'w-', linewidth=2, label="SMPTE 240M Gamut")

# Plot titik primaries + white point
plt.scatter(*red, color='red', label="Red Primary")
plt.scatter(*green, color='green', label="Green Primary")
plt.scatter(*blue, color='blue', label="Blue Primary")
plt.scatter(*white, color='white', edgecolors='black', label="White Point (D65)")

plt.legend(loc="lower right")
plt.title("SMPTE 240M Gamut in CIE 1931 Diagram")
plt.show()