import numpy as np
import matplotlib.pyplot as plt

# ---------- Load CIE 1931 2deg CMFs (local CSV)
# CSV expected: wavelength, xbar, ybar, zbar (header row skipped)
cie = np.loadtxt("CIE_xyz_1931_2deg.csv", delimiter=",", skiprows=1)
xbar = cie[:,1]
ybar = cie[:,2]
zbar = cie[:,3]

# spectral locus xy
sum_xyz = xbar + ybar + zbar
spec_x = xbar / sum_xyz
spec_y = ybar / sum_xyz

# ---------- SMPTE / Rec.709 primaries + white D65
primaries = {
    "red":   (0.64, 0.33),
    "green": (0.30, 0.60),
    "blue":  (0.15, 0.06)
}
white_xy = (0.3127, 0.3290)
tri = np.array([primaries["red"], primaries["green"], primaries["blue"]])

# ---------- Helper: XYZ -> linear sRGB then gamma
M_xyz2srgb = np.array([[ 3.2406, -1.5372, -0.4986],
                       [-0.9689,  1.8758,  0.0415],
                       [ 0.0557, -0.2040,  1.0570]])

def linear_xyz_to_srgb(X, Y, Z):
    """Convert arrays X,Y,Z (shape (N,)) to sRGB values in [0,1]."""
    XYZ = np.vstack([X, Y, Z])  # shape (3, N)
    rgb_lin = M_xyz2srgb.dot(XYZ)  # (3,N)
    # avoid negative values before gamma (set negatives to 0)
    rgb_lin = np.maximum(rgb_lin, 0.0)
    # sRGB companding
    a = 0.0031308
    srgb = np.empty_like(rgb_lin)
    # piecewise function per element
    mask = rgb_lin <= a
    srgb[mask] = 12.92 * rgb_lin[mask]
    srgb[~mask] = 1.055 * np.power(rgb_lin[~mask], 1/2.4) - 0.055
    # clip to [0,1]
    srgb = np.clip(srgb, 0.0, 1.0)
    return srgb.T  # shape (N,3)

# ---------- Rasterize triangle using chromaticity -> color mapping
# Choose bounding box (use full visible region for nicer framing)
xmin, xmax = 0.0, 0.8
ymin, ymax = 0.0, 0.9

W = 800   # horizontal resolution (increase for smoother gradient)
H = 900   # vertical resolution

xs = np.linspace(xmin, xmax, W)
ys = np.linspace(ymin, ymax, H)
grid_x, grid_y = np.meshgrid(xs, ys)  # shape (H, W)

# Vectorized barycentric test to know which pixels lie inside triangle
x0,y0 = tri[0]
x1,y1 = tri[1]
x2,y2 = tri[2]
denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)

s = ((y1 - y2)*(grid_x - x2) + (x2 - x1)*(grid_y - y2)) / denom
t = ((y2 - y0)*(grid_x - x2) + (x0 - x2)*(grid_y - y2)) / denom
u = 1.0 - s - t
inside = (s >= 0) & (t >= 0) & (u >= 0)

# Prepare output image (H, W, 3) and fill inside pixels with computed sRGB
img = np.zeros((H, W, 3), dtype=np.float32)

# For stability, pick a luminance value Y_lum (we use 1.0)
Y_lum = 1.0

# Only compute colors for points inside triangle
inds = np.nonzero(inside)
if inds[0].size > 0:
    px = grid_x[inds]  # x coords of interior pixels (1D)
    py = grid_y[inds]  # y coords of interior pixels (1D)

    # avoid division by zero
    eps = 1e-12
    py_safe = np.where(py <= eps, eps, py)

    # convert xyY -> XYZ with chosen Y luminance
    X_vals = (px / py_safe) * Y_lum
    Y_vals = np.full_like(X_vals, Y_lum)
    Z_vals = ((1.0 - px - py) / py_safe) * Y_lum

    # convert to sRGB (vectorized)
    srgb_vals = linear_xyz_to_srgb(X_vals, Y_vals, Z_vals)  # shape (N,3)

    # write back into image (note imshow origin='lower' later)
    img[inds[0], inds[1], :] = srgb_vals

# ---------- Plotting
fig, ax = plt.subplots(figsize=(7,7))
ax.set_facecolor("black")

# show the rasterized triangle colors. extent maps data coords to axes coords
ax.imshow(img, origin="lower", extent=(xmin, xmax, ymin, ymax), interpolation="bilinear", zorder=0)

# Draw CIE spectral locus (horseshoe) and close loop with straight line (purple line)
# spectral locus drawn in white for similarity with original; you can color it if you like
ax.plot(spec_x, spec_y, color='white', linewidth=1.0, zorder=3)
# close the loop (straight line from last to first)
ax.plot([spec_x[-1], spec_x[0]], [spec_y[-1], spec_y[0]], color='white', linewidth=1.0, zorder=3)

# Draw SMPTE triangle outline on top
tri_closed_x = np.append(tri[:,0], tri[0,0])
tri_closed_y = np.append(tri[:,1], tri[0,1])
ax.plot(tri_closed_x, tri_closed_y, color='white', linewidth=2.0, zorder=4)

# white point marker
ax.plot(white_xy[0], white_xy[1], marker='+', color='white', markersize=12, zorder=5)

# labels & limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("x", color="white")
ax.set_ylabel("y", color="white")
ax.set_title("SMPTE Monitor Gamut (Rec.709) — spectral-colors inside triangle", color="white")
ax.tick_params(colors="white")

plt.show()
