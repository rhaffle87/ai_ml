// gridWorker.js - module worker for heavy grid computation
// Receives message: { type: 'computeGrid', payload: { mMeters, sMeters, gridBounds, nx, ny, simTimeSec } }
// Returns: { type: 'result', maps: [{ masterIndex, slaveIndex, nx, ny, gridBuffer }], contours: [{ masterIndex, slaveIndex, points: [[x,y],...], levelSeconds: 0 }], gridBounds }

// Note: worker does not depend on proj4; it operates in meter coordinates and returns contour points in meter coordinates

function haversineMeters(a, b) {
  // approximate by haversine on lat/lng if a and b have lat/lng, otherwise if they are meter coords, compute Euclidean
  if (a.lat !== undefined && b.lat !== undefined) {
    const R = 6371000;
    const toRad = (d) => d * Math.PI / 180;
    const dLat = toRad(b.lat - a.lat);
    const dLon = toRad(b.lng - a.lng);
    const lat1 = toRad(a.lat);
    const lat2 = toRad(b.lat);
    const h = Math.sin(dLat/2)**2 + Math.cos(lat1)*Math.cos(lat2)*Math.sin(dLon/2)**2;
    return 2 * R * Math.asin(Math.sqrt(h));
  }
  // meter coordinates
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function simulateClockTick(clock, tSec) {
  return (clock && clock.biasSec || 0) + (clock && clock.driftPerSec || 0) * tSec;
}

const C = 299792458;

function computeArrivalSecLocal(station, lat, lng, simTimeSec) {
  // station may have lat/lng or x/y; prefer lat/lng if available
  let dist = 0;
  if (station.lat !== undefined && station.lng !== undefined) {
    dist = haversineMeters({ lat: station.lat, lng: station.lng }, { lat, lng });
  } else if (station.x !== undefined && station.y !== undefined) {
    // convert target lat/lng? In worker we will call with meter coords (x,y) and target x,y
    dist = Math.hypot(station.x - lat, station.y - lng);
  }
  const geoDelay = dist / C;
  const clockOffset = simulateClockTick(station.clock || { biasSec:0, driftPerSec:0 }, simTimeSec) || 0;
  const offsetSec = station.offsetSec || 0;
  let asfMeters = 0;
  // worker does not execute user ASF functions for safety; ASF supplied only as constant avgMeters in diffCorrections or external preprocessing
  if (station.asfMeters !== undefined) asfMeters = station.asfMeters || 0;
  let diffCorrMeters = 0;
  if (station.diffCorrections && station.diffCorrections.enabled) diffCorrMeters = station.diffCorrections.avgMeters || 0;
  return geoDelay + offsetSec + clockOffset + (asfMeters - diffCorrMeters) / C;
}

// Simple marching squares zero-level extraction for a grid of values (tdoa seconds)
function marchingSquaresZero(xs, ys, grid, nx, ny, gridBounds) {
  const lines = [];
  // iterate cells
  for (let j = 0; j < ny-1; j++) {
    for (let i = 0; i < nx-1; i++) {
      const idx00 = j*nx + i;
      const idx10 = j*nx + (i+1);
      const idx01 = (j+1)*nx + i;
      const idx11 = (j+1)*nx + (i+1);
      const v00 = grid[idx00];
      const v10 = grid[idx10];
      const v11 = grid[idx11];
      const v01 = grid[idx01];
      const caseIndex = ((v00>=0)?1:0) | ((v10>=0)?2:0) | ((v11>=0)?4:0) | ((v01>=0)?8:0);
      if (caseIndex === 0 || caseIndex === 15) continue; // no crossing
      // for each edge, compute interpolation point if crossing
      const getX = (ii) => gridBounds.minX + ii * ((gridBounds.maxX - gridBounds.minX)/(nx-1));
      const getY = (jj) => gridBounds.minY + jj * ((gridBounds.maxY - gridBounds.minY)/(ny-1));
      const px = []; const py = [];
      function interp(x1,y1,v1,x2,y2,v2) {
        const t = v1 === v2 ? 0.5 : Math.abs(v1)/(Math.abs(v1)+Math.abs(v2));
        return [ x1 + (x2-x1)*t, y1 + (y2-y1)*t ];
      }
      // edges: 0: left (00-01), 1: top (00-10), 2: right (10-11), 3: bottom (01-11)
      // check each possible edge crossing and push interpolated points in approx order
      if ((v00>=0) !== (v01>=0)) { const p = interp(getX(i), getY(j), v00, getX(i), getY(j+1), v01); px.push(p[0]); py.push(p[1]); }
      if ((v00>=0) !== (v10>=0)) { const p = interp(getX(i), getY(j), v00, getX(i+1), getY(j), v10); px.push(p[0]); py.push(p[1]); }
      if ((v10>=0) !== (v11>=0)) { const p = interp(getX(i+1), getY(j), v10, getX(i+1), getY(j+1), v11); px.push(p[0]); py.push(p[1]); }
      if ((v01>=0) !== (v11>=0)) { const p = interp(getX(i), getY(j+1), v01, getX(i+1), getY(j+1), v11); px.push(p[0]); py.push(p[1]); }
      if (px.length >= 2) {
        const pts = [];
        for (let k=0;k<px.length;k++) pts.push([px[k], py[k]]);
        lines.push(pts);
      }
    }
  }
  return lines;
}

self.onmessage = function(e) {
  const msg = e.data;
  if (!msg || msg.type !== 'computeGrid') return;
  const { mMeters, sMeters, gridBounds, nx, ny, simTimeSec, asfRasters } = msg.payload;
  const nxv = nx, nyv = ny;
  const dx = (gridBounds.maxX - gridBounds.minX) / (nxv - 1);
  const dy = (gridBounds.maxY - gridBounds.minY) / (nyv - 1);

  const maps = [];
  const contours = [];

  // prepare asf raster arrays if provided (parallel to mMeters)
  const asfArrays = (asfRasters && Array.isArray(asfRasters)) ? asfRasters.map(buf => buf ? new Float32Array(buf) : null) : Array(mMeters.length).fill(null);

  for (let mi=0; mi<mMeters.length; mi++){
    for (let si=0; si<sMeters.length; si++){
      const grid = new Float32Array(nxv * nyv);
      // compute pair constant corrections to remove global clock/offset bias so contours reflect geometry
      const masterClock = mMeters[mi].clock || { biasSec: 0, driftPerSec: 0 };
      const slaveClock = sMeters[si].clock || { biasSec: 0, driftPerSec: 0 };
      const masterClockOffset = (masterClock.biasSec || 0) + (masterClock.driftPerSec || 0) * simTimeSec;
      const slaveClockOffset = (slaveClock.biasSec || 0) + (slaveClock.driftPerSec || 0) * simTimeSec;
      const masterOffsetSec = mMeters[mi].offsetSec || 0;
      const slaveOffsetSec = sMeters[si].offsetSec || 0;
      const pairConstSec = (slaveClockOffset + slaveOffsetSec) - (masterClockOffset + masterOffsetSec);
      let idx = 0;
      let nanCount = 0;
      for (let j=0;j<nyv;j++){
        const y = gridBounds.minY + j*dy;
        for (let i=0;i<nxv;i++, idx++){
          const x = gridBounds.minX + i*dx;
          // compute per-station ASF meters from raster if available, otherwise use station.asfMeters
          let asfM = 0;
          if (asfArrays[mi]) asfM = asfArrays[mi][idx] || 0;
          else if (mMeters[mi].asfMeters !== undefined) asfM = mMeters[mi].asfMeters || 0;
          let asfS = 0;
          if (asfArrays[si]) asfS = asfArrays[si][idx] || 0;
          else if (sMeters[si].asfMeters !== undefined) asfS = sMeters[si].asfMeters || 0;

          // compute arrival seconds using local geometry and provided ASF per-point
          // prefer Euclidean distances in meter coordinates (x/y) — do not mix lat/lng with meter points
          const distM = (mMeters[mi].x !== undefined && mMeters[mi].y !== undefined) ? Math.hypot(mMeters[mi].x - x, mMeters[mi].y - y) : ((mMeters[mi].lat !== undefined && mMeters[mi].lng !== undefined) ? haversineMeters({lat:mMeters[mi].lat,lng:mMeters[mi].lng}, {lat:x, lng:y}) : 0);
          const distS = (sMeters[si].x !== undefined && sMeters[si].y !== undefined) ? Math.hypot(sMeters[si].x - x, sMeters[si].y - y) : ((sMeters[si].lat !== undefined && sMeters[si].lng !== undefined) ? haversineMeters({lat:sMeters[si].lat,lng:sMeters[si].lng}, {lat:x, lng:y}) : 0);
          const arrivalM = (distM / C) + (mMeters[mi].offsetSec || 0) + simulateClockTick(mMeters[mi].clock || {biasSec:0,driftPerSec:0}, simTimeSec) + (asfM - (mMeters[mi].diffCorrections && mMeters[mi].diffCorrections.enabled ? (mMeters[mi].diffCorrections.avgMeters||0) : 0)) / C;
          const arrivalS = (distS / C) + (sMeters[si].offsetSec || 0) + simulateClockTick(sMeters[si].clock || {biasSec:0,driftPerSec:0}, simTimeSec) + (asfS - (sMeters[si].diffCorrections && sMeters[si].diffCorrections.enabled ? (sMeters[si].diffCorrections.avgMeters||0) : 0)) / C;
          // remove pair constant clock/offset so contours are based on geometric + ASF/diff variations
          const val = (arrivalS - arrivalM) - pairConstSec; // seconds
          if (!isFinite(val) || Number.isNaN(val)) {
            nanCount++;
            grid[idx] = 0; // safe fallback
          } else {
            grid[idx] = val;
          }
        }
      }
      // marching squares on grid to get zero-contour polylines in meter coords
      const msLines = marchingSquaresZero(null, null, grid, nxv, nyv, gridBounds);
      // each msLine is an array of [x,y] points
      for (const ln of msLines) {
        contours.push({ masterIndex: mi, slaveIndex: si, points: ln, levelSeconds: 0 });
      }
      maps.push({ masterIndex: mi, slaveIndex: si, nx: nxv, ny: nyv, gridBuffer: grid.buffer, nanCount });
    }
  }

  // post result, transferring grid buffers
  self.postMessage({ type: 'result', payload: { maps, contours, gridBounds } }, maps.flatMap(m => [m.gridBuffer]));
};
