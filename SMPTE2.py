import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

# SMPTE (Rec.709) chromaticities
primaries = {
    'red':   (0.64, 0.33),
    'green': (0.30, 0.60),
    'blue':  (0.15, 0.06),
}
white = (0.3127, 0.3290)  # D65

triangle = np.array([primaries['red'], primaries['green'], primaries['blue']])

# Interpolation: generate grid points inside triangle
def inside_triangle(p, tri):
    # barycentric coordinate check
    A = 0.5 * (-tri[1,1]*tri[2,0] + tri[0,1]*(-tri[1,0]+tri[2,0]) + tri[0,0]*(tri[1,1]-tri[2,1]) + tri[1,0]*tri[2,1])
    s = 1/(2*A)*(tri[0,1]*tri[2,0]-tri[0,0]*tri[2,1]+(tri[2,1]-tri[0,1])*p[0]+(tri[0,0]-tri[2,0])*p[1])
    t = 1/(2*A)*(tri[0,0]*tri[1,1]-tri[0,1]*tri[1,0]+(tri[0,1]-tri[1,1])*p[0]+(tri[1,0]-tri[0,0])*p[1])
    return (s >= 0) & (t >= 0) & (1-s-t >= 0), (s, t, 1-s-t)

# Grid resolution
res = 300
xs = np.linspace(0, 0.8, res)
ys = np.linspace(0, 0.9, res)

grid_x, grid_y = np.meshgrid(xs, ys)
rgb_image = np.zeros((res, res, 3))

# Assign colors inside gamut triangle
for i in range(res):
    for j in range(res):
        point = (grid_x[i,j], grid_y[i,j])
        inside, bary = inside_triangle(point, triangle)
        if inside:
            r, g, b = bary
            rgb_image[i,j,:] = [r, g, b]  # mix primaries

# Plotting
fig, ax = plt.subplots(figsize=(6,6))
ax.set_facecolor("black")

# Show filled triangle
ax.imshow(rgb_image, extent=(0,0.8,0,0.9), origin="lower", interpolation="bilinear")

# Draw triangle outline
triangle_closed = np.vstack([triangle, triangle[0]])
ax.plot(triangle_closed[:,0], triangle_closed[:,1], 'w-', linewidth=1.5)

# Plot white point
ax.plot(white[0], white[1], 'w+', markersize=12)

# Axis formatting
ax.set_title("SMPTE Monitor Gamut (Rec.709)", color="white")
ax.set_xlim(0, 0.8)
ax.set_ylim(0, 0.9)
ax.set_xlabel("x", color="white")
ax.set_ylabel("y", color="white")
ax.tick_params(colors="white")
ax.set_aspect('equal')

plt.show()