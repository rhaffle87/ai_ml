import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# CIE 1931 chromaticity diagram boundary (horseshoe shape)
def cie_chromaticity_boundary():
    # Load local dataset
    # Expected format: wavelength, X, Y, Z (columns)
    cie_data = np.loadtxt("CIE_xyz_1931_2deg.csv", delimiter=",", skiprows=1)
    xbar, ybar, zbar = cie_data[:,1], cie_data[:,2], cie_data[:,3]

    # Normalize to get chromaticity coordinates
    X = xbar / (xbar + ybar + zbar)
    Y = ybar / (xbar + ybar + zbar)
    return X, Y

# Simple XYZ → sRGB conversion (linear, clipped)
def xyz_to_rgb(X, Y, Z):
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
    rgb = np.dot(M, np.array([X, Y, Z]))
    return np.clip(rgb, 0, 1)

# SMPTE (Rec. 709) chromaticities
primaries = {
    'red':   (0.64, 0.33),
    'green': (0.30, 0.60),
    'blue':  (0.15, 0.06),
}
white = (0.3127, 0.3290)  # D65
triangle = np.array([primaries['red'], primaries['green'], primaries['blue']])

# Generate filled triangle with rainbow-like blending
res = 300
xs = np.linspace(0, 0.8, res)
ys = np.linspace(0, 0.9, res)
img = np.zeros((res, res, 3))

def inside_triangle(p, tri):
    A = 0.5 * (-tri[1,1]*tri[2,0] + tri[0,1]*(-tri[1,0]+tri[2,0]) + tri[0,0]*(tri[1,1]-tri[2,1]) + tri[1,0]*tri[2,1])
    s = 1/(2*A)*(tri[0,1]*tri[2,0]-tri[0,0]*tri[2,1]+(tri[2,1]-tri[0,1])*p[0]+(tri[0,0]-tri[2,0])*p[1])
    t = 1/(2*A)*(tri[0,0]*tri[1,1]-tri[0,1]*tri[1,0]+(tri[0,1]-tri[1,1])*p[0]+(tri[1,0]-tri[0,0])*p[1])
    return (s >= 0) & (t >= 0) & (1-s-t >= 0), (s, t, 1-s-t)

for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        inside, bary = inside_triangle((x,y), triangle)
        if inside:
            r,g,b = bary
            # Assign RGB according to barycentric weights
            img[res-1-i,j,:] = [r,g,b]

# Create the plot
fig, ax = plt.subplots(figsize=(6,6))
ax.set_facecolor("black")

# Show rainbow triangle
ax.imshow(img, extent=(0,0.8,0,0.9), origin="lower", interpolation="bilinear")

# Plot chromaticity horseshoe boundary and close the loop
X, Y = cie_chromaticity_boundary()
ax.plot(np.append(X, X[0]), np.append(Y, Y[0]), 'w', linewidth=1)

# Plot SMPTE gamut triangle outline
triangle_closed = np.vstack([triangle, triangle[0]])
ax.plot(triangle_closed[:,0], triangle_closed[:,1], 'w-', linewidth=1.5)

# Plot white point
ax.plot(white[0], white[1], 'w+', markersize=12)

# Axis formatting
ax.set_title("SMPTE Monitor Gamut (Rec. 709)", color="white")
ax.set_xlim(0, 0.8)
ax.set_ylim(0, 0.9)
ax.set_xlabel("x", color="white")
ax.set_ylabel("y", color="white")
ax.tick_params(colors="white")
ax.set_aspect('equal')

plt.show()