// Ramer–Douglas–Peucker polyline simplification
// points: array of [x,y]
export default function simplifyRDP(points, epsilon) {
  if (!points || points.length < 3) return points ? points.slice() : [];
  if (!epsilon || epsilon <= 0) return points.slice();

  const sq = (v) => v * v;

  function perpDistance(pt, a, b) {
    // distance from pt to line a-b
    const x = pt[0], y = pt[1];
    const x1 = a[0], y1 = a[1];
    const x2 = b[0], y2 = b[1];
    const dx = x2 - x1; const dy = y2 - y1;
    if (dx === 0 && dy === 0) return Math.hypot(x - x1, y - y1);
    const t = ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy);
    const px = x1 + t * dx; const py = y1 + t * dy;
    return Math.hypot(x - px, y - py);
  }

  const stack = [[0, points.length - 1]];
  const keep = new Uint8Array(points.length);
  keep[0] = 1; keep[points.length - 1] = 1;

  while (stack.length) {
    const [first, last] = stack.pop();
    if (last <= first + 1) continue;
    let maxDist = -1; let index = -1;
    for (let i = first + 1; i < last; i++) {
      const d = perpDistance(points[i], points[first], points[last]);
      if (d > maxDist) { maxDist = d; index = i; }
    }
    if (maxDist > epsilon) {
      keep[index] = 1;
      stack.push([first, index]);
      stack.push([index, last]);
    }
  }

  const out = [];
  for (let i = 0; i < points.length; i++) if (keep[i]) out.push(points[i]);
  return out;
}
