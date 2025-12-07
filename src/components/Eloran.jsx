import React, { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import proj4 from "proj4";
import Papa from "papaparse";
import Modal from './Modal';
import simplifyRDP from '../utils/simplify';
import Waveforms from './Waveforms';

/*
  Eloran Simulator component
  - Built as an extended version of the uploaded Loran-C simulator (see original Loranc.jsx).
  - Adds: DDS emulation, clock types & timing discipline, differential corrections, integrity checks,
          ASF dynamic modeling hooks, GNSS-Eloran fusion toggle.
  - Intended for engineering / classroom / industrial modelling use.
*/

// physical constants
const C = { c: 299792458 };
const TILE_URL_TEMPLATE = import.meta.env.VITE_TILE_URL_TEMPLATE || "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const DEFAULT_FREQ = 100000; // 100 kHz

// small helpers (reused/adapted from original)
function haversine(a, b) {
  const R = 6371000;
  const toRad = (d) => d * Math.PI / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLon = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h = Math.sin(dLat/2)**2 + Math.cos(lat1)*Math.cos(lat2)*Math.sin(dLon/2)**2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
// module-level RNG (seedable) - default to Math.random
let _rng = Math.random;
function mulberry32(a) {
  return function() {
    var t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}
function setRngSeed(seed) {
  if (typeof seed === 'number' && !Number.isNaN(seed)) _rng = mulberry32(seed >>> 0);
  else _rng = Math.random;
}
function gaussianNoise(stdDev) {
  // Box-Muller using module RNG
  let u = 0, v = 0;
  while (u === 0) u = _rng();
  while (v === 0) v = _rng();
  return stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// small clock simulation: bias (s) + drift (s/s)
function simulateClockTick(clock, tSec) {
  // clock = { type: 'local'|'gps'|'cesium', biasSec, driftPerSec }
  // returns effective offset at time tSec
  return (clock.biasSec || 0) + (clock.driftPerSec || 0) * tSec;
}

// compute per-path arrival time (seconds) for a station to a point (lat,lng)
function computeArrivalSec(station, lat, lng, simTimeSec) {
  // station may be master or slave; fields: lat,lng, clock, offsetSec, asfMap, diffCorrections
  const dist = haversine({ lat: station.lat, lng: station.lng }, { lat, lng });
  const geoDelay = dist / C.c;
  const clockOffset = simulateClockTick(station.clock || { biasSec: 0, driftPerSec: 0 }, simTimeSec) || 0;
  const offsetSec = station.offsetSec || 0;
  let asfMeters = 0;
  if (station.asfMap && typeof station.asfMap === 'function') {
    try { asfMeters = station.asfMap(lat, lng) || 0; } catch (e) { asfMeters = 0; }
  }
  let diffCorrMeters = 0;
  if (station.diffCorrections && station.diffCorrections.enabled) diffCorrMeters = (station.diffCorrections.avgMeters || 0);
  return geoDelay + offsetSec + clockOffset + (asfMeters - diffCorrMeters) / C.c;
}

// compute arrival ignoring diffCorrections (used for calibration)
function computeArrivalSecNoDiff(station, lat, lng, simTimeSec) {
  const dist = haversine({ lat: station.lat, lng: station.lng }, { lat, lng });
  const geoDelay = dist / C.c;
  const clockOffset = simulateClockTick(station.clock || { biasSec: 0, driftPerSec: 0 }, simTimeSec) || 0;
  const offsetSec = station.offsetSec || 0;
  let asfMeters = 0;
  if (station.asfMap && typeof station.asfMap === 'function') {
    try { asfMeters = station.asfMap(lat, lng) || 0; } catch (e) { asfMeters = 0; }
  }
  return geoDelay + offsetSec + clockOffset + (asfMeters) / C.c;
}

// build ASF function safely from user input (returns function(lat,lng) -> number)
function createAsfFunctionFromText(code) {
  // Wrap in try/catch by caller; here just construct the function object
  // eslint-disable-next-line no-new-func
  const fn = new Function('lat','lng', code);
  return fn;
}

// integrity evaluator: compute simple protection level (very simplified)
function computeProtectionLevel(errorsMeters) {
  // Use 95% CI approximation: PL = 2 * RMS
  if (!errorsMeters || errorsMeters.length === 0) return 0;
  const mean = errorsMeters.reduce((a,b) => a + b, 0) / errorsMeters.length;
  const sq = errorsMeters.reduce((a,b) => a + (b-mean)*(b-mean), 0) / Math.max(1, errorsMeters.length-1);
  const rms = Math.sqrt(sq + mean*mean);
  return 2 * rms;
}

export default function ELoranSimulator({ tileUrlTemplate = TILE_URL_TEMPLATE }) {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);

  // stations: masters (with additional Eloran fields), slaves, receivers
  const [masters, setMasters] = useState([]); // {lat,lng, txDbm, gri, label, clock: {type,biasSec,driftPerSec}, ddsEnabled, asfMap, diffCorrections}
  const [slaves, setSlaves] = useState([]); // similar to Loran-C slaves
  const [receivers, setReceivers] = useState([]); // {lat,lng,label, fuseMode: 'eLoran'|'GNSS'|'fusion'}
  const [mode, setMode] = useState('add-master');
  const modeRef = useRef(mode);
  const markers = useRef({});
  const estimatedMarkers = useRef({});
  const workerRef = useRef(null);
  const gridMapsRef = useRef(null); // store large Float32Array maps here to avoid storing in state

  // simulation & status
  const [gridStatus, setGridStatus] = useState(null);
  const [simulationResults, setSimulationResults] = useState(null);
  const [logEvents, setLogEvents] = useState([]); // DDS logs, integrity events
  const [enableDDSGlobal, setEnableDDSGlobal] = useState(true);
  const [enableIntegrityChecks, setEnableIntegrityChecks] = useState(true);
  const [integrityThresholdMeters, setIntegrityThresholdMeters] = useState(50); // simple user threshold
  const [timeSinceStart, setTimeSinceStart] = useState(0); // simulation time (s)
  const simTimeRef = useRef(0);
  const [asfText, setAsfText] = useState('return 0;');
  const [asfTarget, setAsfTarget] = useState('');
  const [contourSimplifyToleranceMeters, setContourSimplifyToleranceMeters] = useState(8);
  const [samplerConcurrency, setSamplerConcurrency] = useState(3);
  // modal state for non-blocking prompt/confirm
  const [modalState, setModalState] = useState({ open: false, type: 'info', title: '', message: '' });
  const [modalInput, setModalInput] = useState('');
  const modalResolveRef = useRef(null);
  const [contourUnit, setContourUnit] = useState('meters'); // 'meters' or 'seconds'
  const asfWorkerRef = useRef(null);
  const [estimatorMode, setEstimatorMode] = useState('controlled'); // 'controlled'|'random'|'none'
  const [estNoiseStdMeters, setEstNoiseStdMeters] = useState(20);
  const [rngSeed, setRngSeedState] = useState('');
  // load persisted seed on mount
  useEffect(() => {
    try {
      const s = localStorage.getItem('eloran_rng_seed');
      if (s) {
        const si = parseInt(s);
        if (!Number.isNaN(si)) { setRngSeed(si); setRngSeedState(String(si)); }
      }
    } catch (e) {}
  }, []);

  // show a prompt modal, returns Promise<string|null>
  function showPrompt(title, defaultValue='') {
    return new Promise((resolve) => {
      modalResolveRef.current = resolve;
      setModalInput(defaultValue);
      setModalState({ open: true, type: 'prompt', title, message: '' });
    });
  }

  // show a confirm modal, returns Promise<boolean>
  function showConfirm(message, title='Confirm') {
    return new Promise((resolve) => {
      modalResolveRef.current = resolve;
      setModalInput('');
      setModalState({ open: true, type: 'confirm', title, message });
    });
  }

  function handleModalCancel() {
    try { if (modalResolveRef.current) modalResolveRef.current(modalState.type === 'prompt' ? null : false); } catch(e) {}
    modalResolveRef.current = null;
    setModalState({ open: false, type: 'info', title: '', message: '' });
  }

  function handleModalConfirm(val) {
    try { if (modalResolveRef.current) modalResolveRef.current(modalState.type === 'prompt' ? val : true); } catch(e) {}
    modalResolveRef.current = null;
    setModalState({ open: false, type: 'info', title: '', message: '' });
  }
  const [detectJitterMs, setDetectJitterMs] = useState(1);
  const [skyEnabled, setSkyEnabled] = useState(false);
  const [skyDelayMs, setSkyDelayMs] = useState(1);
  const [skyAmpFraction, setSkyAmpFraction] = useState(0.3);
  const [recentErrors, setRecentErrors] = useState([]);
  const [recentHPLs, setRecentHPLs] = useState([]);
  // lightweight toast notifications
  const [toasts, setToasts] = useState([]);
  const toastId = useRef(0);
  function showToast(msg, type = 'info', ttl = 4000) {
    const id = ++toastId.current;
    setToasts(prev => [...prev, { id, msg, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), ttl);
  }

  // counters
  const masterCounter = useRef(0);
  const slaveCounter = useRef(0);
  const receiverCounter = useRef(0);

  // create map
  useEffect(() => {
    mapRef.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'raster-tiles': {
            type: 'raster',
            tiles: [tileUrlTemplate],
            tileSize: 256
          }
        },
        layers: [{ id: 'simple-tiles', type: 'raster', source: 'raster-tiles' }]
      },
      center: [106.816666, -6.200000],
      zoom: 5
    });

    mapRef.current.addControl(new maplibregl.NavigationControl());

    // click handlers to place markers
    mapRef.current.on('click', (e) => {
      const lnglat = e.lngLat;
      if (modeRef.current === 'add-master') addMaster({ lat: lnglat.lat, lng: lnglat.lng });
      else if (modeRef.current === 'add-slave') addSlave({ lat: lnglat.lat, lng: lnglat.lng });
      else if (modeRef.current === 'add-receiver') addReceiver({ lat: lnglat.lat, lng: lnglat.lng });
    });

    // cleanup
    return () => { if (mapRef.current) mapRef.current.remove();
      try { if (workerRef.current) { workerRef.current.terminate(); workerRef.current = null; } } catch(e) {}
      try { if (asfWorkerRef.current) { asfWorkerRef.current.terminate(); asfWorkerRef.current = null; } } catch(e) {}
    };
    // eslint-disable-next-line
  }, []);

  useEffect(()=> { modeRef.current = mode; }, [mode]);

  // marker adders: reuse style but with Eloran badges
  const addMarker = useCallback((point, label, type, meta={}) => {
    if (!mapRef.current) return;
    const el = document.createElement('div');
    el.className = `marker marker-${type}`;
    el.style.display = 'flex';
    el.style.flexDirection = 'column';
    el.style.alignItems = 'center';
    el.style.height = '24px';
    el.style.overflow = 'visible';

    const dot = document.createElement('div');
    dot.style.width = '20px';
    dot.style.height = '20px';
    dot.style.borderRadius = '50%';
    if (type === 'master') dot.style.background = '#0ea5a4';
    if (type === 'slave') dot.style.background = '#f59e0b';
    if (type === 'receiver') dot.style.background = '#1e3a8a';

    const text = document.createElement('div');
    text.innerText = label;
    text.style.fontSize = '11px';
    text.style.color = 'black';
    text.style.position = 'absolute';
    text.style.top = '22px';
    text.style.left = '50%';
    text.style.transform = 'translateX(-50%)';
    text.style.textShadow = '0 1px 2px rgba(255,255,255,0.8)';

    el.appendChild(dot);
    el.appendChild(text);

    const marker = new maplibregl.Marker({ element: el, anchor: 'center', draggable: true })
      .setLngLat([point.lng, point.lat])
      .addTo(mapRef.current);

    marker.on('dragend', () => {
      const lnglat = marker.getLngLat();
      if (type === 'master') setMasters(prev => prev.map(p => p.label === label ? {...p, lat: lnglat.lat, lng: lnglat.lng} : p));
      if (type === 'slave') setSlaves(prev => prev.map(p => p.label === label ? {...p, lat: lnglat.lat, lng: lnglat.lng} : p));
      if (type === 'receiver') setReceivers(prev => prev.map(p => p.label === label ? {...p, lat: lnglat.lat, lng: lnglat.lng} : p));
    });

    el.addEventListener('click', (e) => {
      e.stopPropagation();
      if (modeRef.current === 'pan') {
        marker.remove();
        delete markers.current[label];
        if (type === 'master') setMasters(prev => prev.filter(m => m.label !== label));
        if (type === 'slave') setSlaves(prev => prev.filter(s => s.label !== label));
        if (type === 'receiver') setReceivers(prev => prev.filter(r => r.label !== label));
      }
    });

    markers.current[label] = marker;
  }, []);

  const addMaster = useCallback((point) => {
    masterCounter.current++;
    const label = `M${masterCounter.current}`;
    const m = {
      ...point,
      txDbm: 20,
      gri: 8330,
      label,
      // default Eloran additions
      clock: { type: 'gps-disciplined', biasSec: 0, driftPerSec: 0 },
      ddsEnabled: true,
      asfMap: null, // user-editable
      diffCorrections: { enabled: true, avgMeters: 0 },
    };
    setMasters(prev => [...prev, m]);
    addMarker(point, label, 'master');
  }, [addMarker]);

  const addSlave = useCallback((point) => {
    slaveCounter.current++;
    const label = `S${slaveCounter.current}`;
    const s = { ...point, txDbm: 18, offsetSec: 0, label };
    setSlaves(prev => [...prev, s]);
    addMarker(point, label, 'slave');
  }, [addMarker]);

  const addReceiver = useCallback((point) => {
    receiverCounter.current++;
    const label = `R${receiverCounter.current}`;
    const r = { ...point, label, fuseMode: 'eLoran', lastFix: null };
    setReceivers(prev => [...prev, r]);
    addMarker(point, label, 'receiver');
  }, [addMarker]);

  // CSV import (extended to accept Eloran fields if present)
  const handleCsvImport = useCallback((event) => {
    const file = event.target.files[0];
    if (!file) return;
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const data = results.data;
        let validRows = 0, errors = [];
        data.forEach((row, index) => {
          const role = (row.role || '').toLowerCase().trim();
          const lat = parseFloat(row.lat);
          const lng = parseFloat(row.lng);
          if (!role || !['master','slave','receiver'].includes(role)) {
            errors.push(`Row ${index+1}: invalid role '${row.role}'`);
            return;
          }
          if (isNaN(lat) || isNaN(lng)) {
            errors.push(`Row ${index+1}: invalid coords`);
            return;
          }
          const point = { lat, lng };
          if (role === 'master') {
            masterCounter.current++;
            const label = row.label || `M${masterCounter.current}`;
            const m = {
              ...point,
              txDbm: parseFloat(row.txDbm) || 20,
              gri: row.gri || 8330,
              label,
              clock: { type: row.clockType || 'gps-disciplined', biasSec: parseFloat(row.clockBias||0), driftPerSec: parseFloat(row.clockDrift||0) },
              ddsEnabled: (row.ddsEnabled === 'false') ? false : true,
              asfMap: null,
              diffCorrections: { enabled: true, avgMeters: 0 }
            };
            setMasters(prev => [...prev, m]);
            addMarker(point, label, 'master');
          } else if (role === 'slave') {
            slaveCounter.current++;
            const label = row.label || `S${slaveCounter.current}`;
            const s = { ...point, txDbm: parseFloat(row.txDbm)||18, offsetSec: parseFloat(row.offsetSec)||0, label };
            setSlaves(prev => [...prev, s]);
            addMarker(point, label, 'slave');
          } else {
            receiverCounter.current++;
            const label = row.label || `R${receiverCounter.current}`;
            const r = { ...point, label, fuseMode: 'eLoran', lastFix: null };
            setReceivers(prev => [...prev, r]);
            addMarker(point, label, 'receiver');
          }
          validRows++;
        });
        if (errors.length) {
          showToast(`Import completed: ${validRows} added, errors: ${errors.join('; ')}`, 'error', 8000);
        } else {
          showToast(`Import successful: ${validRows} stations added`, 'success', 4000);
        }
      },
      error: (err) => { showToast(`Parsing error: ${err.message}`, 'error', 5000); }
    });
    event.target.value = '';
  }, [addMarker]);

  // --- Eloran core: compute TDOA grid like Loran-C but apply ASF + diff corrections and DDS timing ---
  async function computeGrid(nx=200, ny=200) {
    if (masters.length === 0 || slaves.length === 0) { showToast('Add at least one master and one slave', 'error'); return; }
    // prepare grid bounds from station extents (EPSG:3857)
    const all = [...masters, ...slaves, ...receivers];
    const lats = all.map(a => a.lat), lngs = all.map(a => a.lng);
    let bbox = [Math.min(...lngs), Math.min(...lats), Math.max(...lngs), Math.max(...lats)];
    // protect against degenerate bbox (all stations at same coord)
    if (Math.abs(bbox[2] - bbox[0]) < 1e-6) {
      bbox[0] = bbox[0] - 0.01; bbox[2] = bbox[2] + 0.01;
    }
    if (Math.abs(bbox[3] - bbox[1]) < 1e-6) {
      bbox[1] = bbox[1] - 0.01; bbox[3] = bbox[3] + 0.01;
    }
    const padX = Math.max((bbox[2]-bbox[0]) * 0.35, 0.01), padY = Math.max((bbox[3]-bbox[1]) * 0.35, 0.01);
    const bboxP = [bbox[0]-padX, bbox[1]-padY, bbox[2]+padX, bbox[3]+padY];
    const bl = proj4('EPSG:4326','EPSG:3857',[bboxP[0], bboxP[1]]);
    const tr = proj4('EPSG:4326','EPSG:3857',[bboxP[2], bboxP[3]]);
    const gridBounds = { minX: bl[0], minY: bl[1], maxX: tr[0], maxY: tr[1] };

    // convert masters/slaves to meter coords and attach ASF/diff information
    const mMeters = masters.map(m => {
      const xy = proj4('EPSG:4326','EPSG:3857',[m.lng, m.lat]);
      return { x: xy[0], y: xy[1], lat: m.lat, lng: m.lng, label: m.label, asfMap: m.asfMap, clock: m.clock, ddsEnabled: m.ddsEnabled, diffCorrections: m.diffCorrections };
    });
    const sMeters = slaves.map(s => {
      const xy = proj4('EPSG:4326','EPSG:3857',[s.lng, s.lat]);
      return { x: xy[0], y: xy[1], lat: s.lat, lng: s.lng, label: s.label, offsetSec: s.offsetSec || 0, diffCorrections: s.diffCorrections, asfMap: s.asfMap };
    });
    const rMeters = receivers.map(r => {
      const xy = proj4('EPSG:4326','EPSG:3857',[r.lng, r.lat]);
      return { x: xy[0], y: xy[1], label: r.label, lat: r.lat, lng: r.lng, fuseMode: r.fuseMode };
    });

    // compute baseline distances to set sensible contour levels
    let maxDist = 0;
    for (let mi=0; mi<mMeters.length; mi++){
      for (let si=0; si<sMeters.length; si++){
        const d = Math.hypot(mMeters[mi].x - sMeters[si].x, mMeters[mi].y - sMeters[si].y);
        if (d > maxDist) maxDist = d;
      }
    }
    const step = Math.max(100, maxDist / 20);
    const levelsMeters = [];
    for (let k = -10; k <= 10; k++) levelsMeters.push(k * step);

    // We will compute TDOA grid client-side (lightweight version). We add ASF and diffCorrections adjustments:
    // - ASF: if master.asfMap provided, look up approximate asf (meters) for cell center (stubbed)
    // - diffCorrections: if enabled, subtract avg correction (meters)
    setGridStatus({ status: 'computing', nx, ny });

    // prepare station copies for worker; include lat/lng and constant asfMeters if asfMap is not a function
    const mForWorker = mMeters.map(m => ({ x: m.x, y: m.y, lat: m.lat, lng: m.lng, clock: m.clock, diffCorrections: m.diffCorrections, offsetSec: 0, asfMeters: (typeof masters.find(mm=>mm.label===m.label)?.asfMap === 'number' ? masters.find(mm=>mm.label===m.label).asfMap : undefined) }));
    const sForWorker = sMeters.map(s => ({ x: s.x, y: s.y, lat: s.lat, lng: s.lng, clock: s.clock, diffCorrections: s.diffCorrections, offsetSec: s.offsetSec || 0, asfMeters: undefined }));

    // if any master has a function asfMap, pre-sample it into rasters so the grid worker never needs to eval functions
    const hasFunctionAsf = masters.some(m => m.asfMap && typeof m.asfMap === 'function');

    // helper to sample rasters for a function-based ASF using multiple short-lived workers (bounded concurrency)
    async function sampleAsfRastersForMasters(masterList, concurrency = 3) {
      // prepare lat/lng arrays for every grid cell (cell centers) — reuse for all masters
      const nxv = nx, nyv = ny;
      const dx = (gridBounds.maxX - gridBounds.minX) / (nxv - 1);
      const dy = (gridBounds.maxY - gridBounds.minY) / (nyv - 1);
      const latArr = new Float64Array(nxv * nyv);
      const lngArr = new Float64Array(nxv * nyv);
      let idx = 0;
      for (let j = 0; j < nyv; j++) {
        const y = gridBounds.minY + j * dy;
        for (let i = 0; i < nxv; i++, idx++) {
          const x = gridBounds.minX + i * dx;
          const [lngc, latc] = proj4('EPSG:3857','EPSG:4326',[x,y]);
          latArr[idx] = latc; lngArr[idx] = lngc;
        }
      }

      const rasters = Array(masterList.length).fill(null);
      const tasks = [];
      for (let mi = 0; mi < masterList.length; mi++) {
        const m = masterList[mi];
        if (!m.asfMap || typeof m.asfMap !== 'function') continue;
        const code = 'return (' + m.asfMap.toString() + ')(lat,lng);';
        tasks.push({ mi, code });
      }

      if (tasks.length === 0) return rasters;

      // concurrency-limited runner
      let nextTask = 0;
      async function runTask(task) {
        const { mi, code } = task;
        return new Promise((resolve) => {
          const w = new Worker(new URL('../workers/asfWorker.js', import.meta.url), { type: 'module' });
          const tid = setTimeout(() => {
            try { w.terminate(); } catch(e) {}
            console.warn('ASF sampling timeout for master index', mi);
            resolve({ mi, arr: null });
          }, 12000);
          const onmsg = (ev) => {
            const mm = ev.data;
            if (!mm) return;
            if (mm.type === 'result' && mm.payload && mm.payload.buffer) {
              clearTimeout(tid);
              w.removeEventListener('message', onmsg);
              const arr = new Float32Array(mm.payload.buffer);
              try { w.terminate(); } catch(e) {}
              resolve({ mi, arr });
            } else if (mm.type === 'error') {
              clearTimeout(tid);
              w.removeEventListener('message', onmsg);
              try { w.terminate(); } catch(e) {}
              console.warn('ASF sampling error for master', mi, mm.payload && mm.payload.message);
              resolve({ mi, arr: null });
            }
          };
          w.addEventListener('message', onmsg);
          try {
            w.postMessage({ type: 'sampleBatch', payload: { code, lats: latArr, lngs: lngArr, nx: nxv, ny: nyv } });
          } catch (err) {
            clearTimeout(tid);
            w.removeEventListener('message', onmsg);
            try { w.terminate(); } catch(e) {}
            console.warn('ASF sampling postMessage failed for master', mi, err);
            resolve({ mi, arr: null });
          }
        });
      }

      // worker pool loop
      const results = [];
      const runners = [];
      const concurrencyLimit = Math.max(1, Math.min(concurrency, tasks.length));
      for (let i = 0; i < concurrencyLimit; i++) {
        runners.push((async function workerLoop() {
          while (nextTask < tasks.length) {
            const task = tasks[nextTask++];
            // eslint-disable-next-line no-await-in-loop
            const res = await runTask(task);
            results.push(res);
          }
        })());
      }
      await Promise.all(runners);

      // populate rasters by master index
      for (const r of results) {
        rasters[r.mi] = r.arr;
      }
      return rasters;
    }

    // start or reuse grid worker
    try {
      if (!workerRef.current) {
        workerRef.current = new Worker(new URL('../workers/gridWorker.js', import.meta.url), { type: 'module' });
      }
      const w = workerRef.current;
      w.onmessage = (ev) => {
        const msg = ev.data;
        if (!msg) return;
        if (msg.type === 'result') {
              const { maps, contours: contoursMeters, gridBounds: gb } = msg.payload;
              const contours = contoursMeters.map(c => ({ masterIndex: c.masterIndex, slaveIndex: c.slaveIndex, points: c.points, levelSeconds: c.levelSeconds }));
              // Move large grid buffers to ref to avoid expensive React state copies
              const mapsConverted = maps.map(m => ({ masterIndex: m.masterIndex, slaveIndex: m.slaveIndex, nx: m.nx, ny: m.ny, gridBounds: gb, data: new Float32Array(m.gridBuffer), units: 'seconds' }));
              gridMapsRef.current = mapsConverted;
              setGridStatus({ status: 'ready', computedAt: Date.now(), mapsCount: mapsConverted.length, contours, gridBounds: gb, gridUnits: 'seconds' });
              showToast(`Grid computed: ${mapsConverted.length} maps, ${contours.length} contours`, 'success', 3000);
              drawLOPs(contours);
          }
      };

      // if function ASFs exist, pre-sample them into rasters and include them in payload
      let asfRasters = null;
      if (hasFunctionAsf) {
        const ras = await sampleAsfRastersForMasters(mMeters, samplerConcurrency);
        // convert to ArrayBuffer list (null where not present)
        asfRasters = ras.map(a => a ? a.buffer : null);
      }

      // prepare transfer list with asf rasters (if any)
      const transfer = [];
      if (asfRasters) asfRasters.forEach(b => { if (b) transfer.push(b); });
      w.postMessage({ type: 'computeGrid', payload: { mMeters: mForWorker, sMeters: sForWorker, gridBounds, nx, ny, simTimeSec: simTimeRef.current, asfRasters } }, transfer);
      return;
    } catch (err) {
      // fallback to synchronous compute if worker fails
      console.error('Worker failed, falling back to main-thread compute', err);
    }

    // naive synchronous compute (for clarity) — acceptable for moderate grid sizes
    try {
      const nxv = nx, nyv = ny;
      const xs = new Float64Array(nxv), ys = new Float64Array(nyv);
      const dx = (gridBounds.maxX - gridBounds.minX) / (nxv - 1);
      const dy = (gridBounds.maxY - gridBounds.minY) / (nyv - 1);
      for (let i=0;i<nxv;i++) xs[i] = gridBounds.minX + i*dx;
      for (let j=0;j<nyv;j++) ys[j] = gridBounds.minY + j*dy;

      const maps = [];
      const contours = []; // simplified: collect zero-level hyperbolas only (for visualization)
      for (let mi=0; mi<mMeters.length; mi++){
        for (let si=0; si<sMeters.length; si++){
          const grid = new Float32Array(nxv * nyv);
          // compute per-pair constant (clock + offset) to normalize grid and keep zero-crossings
          const mClock = mMeters[mi].clock || { biasSec: 0, driftPerSec: 0 };
          const sClock = sMeters[si].clock || { biasSec: 0, driftPerSec: 0 };
          const mClockOffset = (mClock.biasSec || 0) + (mClock.driftPerSec || 0) * simTimeRef.current;
          const sClockOffset = (sClock.biasSec || 0) + (sClock.driftPerSec || 0) * simTimeRef.current;
          const mOffset = mMeters[mi].offsetSec || 0;
          const sOffset = sMeters[si].offsetSec || 0;
          const pairConstSec = (sClockOffset + sOffset) - (mClockOffset + mOffset);
          let idx = 0;
          for (let j=0;j<nyv;j++){
            const y = ys[j];
            for (let i=0;i<nxv;i++, idx++){
              const x = xs[i];
              const [lngc, latc] = proj4('EPSG:3857','EPSG:4326',[x,y]);
              const arrivalM_sec = computeArrivalSec(mMeters[mi], latc, lngc, simTimeRef.current);
              const arrivalS_sec = computeArrivalSec(sMeters[si], latc, lngc, simTimeRef.current);

              // store the grid as TDOA in seconds (arrivalS - arrivalM), with per-pair constant removed
              grid[idx] = (arrivalS_sec - arrivalM_sec) - pairConstSec;
            }
          }
          maps.push({ masterIndex: mi, slaveIndex: si, nx: nxv, ny: nyv, gridBounds, data: grid.buffer });
          // for contours, extract quick zero crossings (very simple marching algorithm for visualization)
          // we'll sample row-wise and push connecting points where sign changes (fast and coarse)
          const poly = [];
          for (let j=0;j<nyv;j++){
            for (let i=0;i<nxv-1;i++){
              const a = grid[j*nxv + i], b = grid[j*nxv + i+1];
              if (a === 0 || b === 0 || (a<0 && b>0) || (a>0 && b<0)) {
                const x0 = gridBounds.minX + i*dx;
                const y0 = gridBounds.minY + j*dy;
                poly.push([x0, y0]);
              }
            }
          }
          if (poly.length>1) contours.push({ masterIndex: mi, slaveIndex: si, points: poly, levelSeconds: 0 });
        }
      }

      // finalize
      // convert map buffers to Float32Array for downstream use
      const mapsConverted = maps.map(m => ({ masterIndex: m.masterIndex, slaveIndex: m.slaveIndex, nx: m.nx, ny: m.ny, gridBounds: m.gridBounds, data: new Float32Array(m.data), units: 'seconds' }));
      gridMapsRef.current = mapsConverted;
      setGridStatus({ status: 'ready', computedAt: Date.now(), mapsCount: mapsConverted.length, contours, gridBounds, gridUnits: 'seconds' });
      // draw LOPs
      drawLOPs(contours);
    } catch (err) {
      setGridStatus({ status: 'error', message: String(err) });
    }
  }

  // draw LOPs (similar to Loranc.jsx)
  function drawLOPs(contours) {
    if (!mapRef.current) return;
    try {
      if (mapRef.current.getLayer('elops')) mapRef.current.removeLayer('elops');
      if (mapRef.current.getSource('elops')) mapRef.current.removeSource('elops');
    } catch(e) { /* ignore */ }

    const features = [];
    (contours || []).forEach((contour, idx) => {
      try {
        let pts = contour.points || [];
        if (contourSimplifyToleranceMeters && contourSimplifyToleranceMeters > 0) {
          try { pts = simplifyRDP(pts, contourSimplifyToleranceMeters); } catch(e) { pts = contour.points || []; }
        }
        if (!pts || pts.length < 2) return; // need at least two points for LineString
        const coords = pts.map(([x,y]) => proj4('EPSG:3857','EPSG:4326',[x,y]));
        if (!coords || coords.length < 2) return;
        const levelSeconds = typeof contour.levelSeconds !== 'undefined' ? contour.levelSeconds : 0;
        const levelMeters = levelSeconds * C.c;
        features.push({
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: coords },
          properties: { id: `elop-${contour.masterIndex}-${contour.slaveIndex}-${idx}`, masterIndex: contour.masterIndex, slaveIndex: contour.slaveIndex, levelSeconds, levelMeters }
        });
      } catch (err) {
        console.warn('Failed to process contour', err);
      }
    });

    if (features.length === 0) {
      showToast('No contours generated (check grid resolution / ASF sampling)', 'error', 4000);
      return;
    }
    const geojson = { type: 'FeatureCollection', features };
    try {
      mapRef.current.addSource('elops', { type: 'geojson', data: geojson });
      mapRef.current.addLayer({ id: 'elops', type: 'line', source: 'elops', paint: { 'line-width': 2, 'line-opacity': 0.9, 'line-color': '#06b6d4' } });
    } catch (err) {
      console.error('Failed to add elops source/layer', err);
      showToast('Failed to render contours: ' + (err && err.message ? err.message : String(err)), 'error', 5000);
      return;
    }
    // generate label features at centroid of each contour
    if (mapRef.current.getLayer('elop-labels')) mapRef.current.removeLayer('elop-labels');
    if (mapRef.current.getSource('elop-labels')) mapRef.current.removeSource('elop-labels');
    const labelFeatures = features.map(f => {
      const coords = f.geometry.coordinates;
      const cx = coords.reduce((s,c)=>s+c[0],0)/coords.length;
      const cy = coords.reduce((s,c)=>s+c[1],0)/coords.length;
      // compute display text according to contourUnit
      const sec = f.properties.levelSeconds || 0;
      const meters = f.properties.levelMeters || (sec * C.c);
      const labelText = contourUnit === 'meters' ? `${meters.toFixed(2)} m` : `${sec.toExponential(3)} s`;
      return { type: 'Feature', geometry: { type: 'Point', coordinates: [cx, cy] }, properties: { labelText } };
    });
    const labelsGeo = { type: 'FeatureCollection', features: labelFeatures };
    mapRef.current.addSource('elop-labels', { type: 'geojson', data: labelsGeo });
    mapRef.current.addLayer({
      id: 'elop-labels',
      type: 'symbol',
      source: 'elop-labels',
      layout: {
        'text-field': ['get', 'labelText'],
        'text-size': 12,
        'text-offset': [0, -0.6]
      },
      paint: { 'text-color': '#064e3b' }
    });

    // remove previous click handler if any
    try { mapRef.current.off('click', 'elops'); } catch(e) {}
    // show popup on LOP click with seconds/meters info
    mapRef.current.on('click', 'elops', (e) => {
      const feat = e.features && e.features[0];
      if (!feat) return;
      const props = feat.properties || {};
      const sec = parseFloat(props.levelSeconds || 0);
      const m = parseFloat(props.levelMeters || (sec * C.c));
      const html = `<div style="font-size:12px"><strong>LOP</strong><br/>Master: ${props.masterIndex}, Slave: ${props.slaveIndex}<br/>TDOA: ${sec.toExponential(3)} s<br/>≈ ${m.toFixed(3)} m</div>`;
      new maplibregl.Popup().setLngLat(e.lngLat).setHTML(html).addTo(mapRef.current);
    });
  }

  // simulate DDS broadcast from masters: create message objects with UTC, integrity, diffs
  function broadcastDDS(nowSec) {
    if (!enableDDSGlobal) return;
    const events = [];
    masters.forEach((m) => {
      if (!m.ddsEnabled) return;
      // message content (simplified): { seq, utc, clockBias, integrityStatus, diffCorrectionMeters }
      const seq = Math.floor(_rng()*100000);
      const utcApprox = Date.now() + ((m.clock && m.clock.biasSec) ? m.clock.biasSec * 1000 : 0);
      const integrity = enableIntegrityChecks ? 'OK' : 'UNKNOWN';
      const diff = m.diffCorrections && m.diffCorrections.enabled ? (m.diffCorrections.avgMeters || 0) : 0;
      const msg = { from: m.label, seq, utcMs: utcApprox, integrity, diffMeters: diff, timestampSimSec: nowSec };
      events.push({ type: 'DDS', msg, station: m.label, time: Date.now() });
    });
    if (events.length) {
      setLogEvents(prev => [...prev.slice(-400), ...events]); // keep buffer
    }
  }

  // Simulated receiver estimation using TDOA + differential corrections + optional GNSS fusion
  function estimateReceiver(receiverIndex=0) {
    if (receivers.length === 0) return;
    const rx = receivers[receiverIndex];
    const refMaster = masters[0];
    if (!refMaster) { showToast('Add a master station first for TDOA reference', 'error'); return; }

    // build pairs with slaves using per-path arrival times (via helper)
    const pairs = [];
    for (let si=0; si<slaves.length; si++){
      const s = slaves[si];
      const arrivalM = computeArrivalSec(refMaster, rx.lat, rx.lng, simTimeRef.current);
      const arrivalS = computeArrivalSec(s, rx.lat, rx.lng, simTimeRef.current);
      const tdoaSec = arrivalS - arrivalM; // slave - master
      pairs.push({ master: refMaster, slave: s, tdoaSec });
    }

    // initial guess: use receiver location as deterministic start (no random perturbation)
    const initialGuess = { lat: rx.lat, lng: rx.lng };
    const estObj = solvePositionFromTDOA(pairs, initialGuess);
    const est = { lat: estObj.lat, lng: estObj.lng };

    // fuse with GNSS if requested (very simple weighted average)
    let fused = est;
    // track covariance and HPL from solver (meters)
    let estCov = estObj.covariance || [[0,0],[0,0]];
    // estObj.hplMeters may be undefined when solver couldn't compute covariance; use null to indicate unavailable
    let estHpl = (typeof estObj.hplMeters === 'number') ? estObj.hplMeters : null;
    if (rx.fuseMode === 'fusion') {
      // synthesize GNSS fix as true + gaussian error (5-10 m)
      // use a deterministic small GNSS offset to keep estimations repeatable
      const gnssErrorMeters = 8; // deterministic nominal GNSS error
      const latOffset = (gnssErrorMeters / 111320) * 0.0; // no random lateral bias
      const lngOffset = (gnssErrorMeters / (111320 * Math.cos(rx.lat * Math.PI/180))) * 0.0;
      const gnssFix = { lat: rx.lat + latOffset, lng: rx.lng + lngOffset };
      // weighted average: give Eloran 0.6 weight when diffs present
      const wE = 0.6, wG = 0.4;
      fused = { lat: est.lat * wE + gnssFix.lat * wG, lng: est.lng * wE + gnssFix.lng * wG };
      // combine covariance simplistically: fusedCov = wE^2 * estCov + wG^2 * gnssCov
      const gnssSigma = 8; const gnssCov = [[gnssSigma*gnssSigma,0],[0,gnssSigma*gnssSigma]];
      estCov = [[ wE*wE*estCov[0][0] + wG*wG*gnssCov[0][0], wE*wE*estCov[0][1] + wG*wG*gnssCov[0][1] ],
                [ wE*wE*estCov[1][0] + wG*wG*gnssCov[1][0], wE*wE*estCov[1][1] + wG*wG*gnssCov[1][1] ]];
      // recompute HPL from estCov
      const trace = estCov[0][0] + estCov[1][1];
      const det = estCov[0][0]*estCov[1][1] - estCov[0][1]*estCov[1][0];
      const temp = Math.sqrt(Math.max(0, (trace*trace)/4 - det));
      const lambda1 = Math.max(0, trace/2 + temp);
      estHpl = 3 * Math.sqrt(lambda1);
    }

    // apply estimator mode: 'controlled' = Gaussian noise with configured σ,
    // 'random' = Gaussian noise with variable amplitude, 'none' = no noise
    if (estimatorMode === 'controlled') {
      const noiseLatMeters = gaussianNoise(estNoiseStdMeters);
      const noiseLngMeters = gaussianNoise(estNoiseStdMeters);
      const latOffsetDeg = noiseLatMeters / 111320;
      const lngOffsetDeg = noiseLngMeters / (111320 * Math.cos(rx.lat * Math.PI/180));
      fused = { lat: fused.lat + latOffsetDeg, lng: fused.lng + lngOffsetDeg };
    } else if (estimatorMode === 'random') {
      // variable amplitude noise: random scale between 0.5x and 3x the configured σ
      const randomScale = 0.5 + _rng() * 2.5;
      const noiseLatMeters = gaussianNoise(estNoiseStdMeters * randomScale);
      const noiseLngMeters = gaussianNoise(estNoiseStdMeters * randomScale);
      const latOffsetDeg = noiseLatMeters / 111320;
      const lngOffsetDeg = noiseLngMeters / (111320 * Math.cos(rx.lat * Math.PI/180));
      fused = { lat: fused.lat + latOffsetDeg, lng: fused.lng + lngOffsetDeg };
    }

    // record estimated marker
    if (estimatedMarkers.current[receiverIndex]) {
      estimatedMarkers.current[receiverIndex].remove();
      delete estimatedMarkers.current[receiverIndex];
    }
    const el = document.createElement('div');
    el.className = 'marker marker-est';
    el.style.display = 'flex';
    el.style.flexDirection = 'column';
    el.style.alignItems = 'center';
    const dot = document.createElement('div');
    dot.style.width = '16px'; dot.style.height='16px'; dot.style.borderRadius='50%'; dot.style.background='#333';
    const text = document.createElement('div'); text.innerText = `Est ${rx.label}`; text.style.fontSize='10px';
    el.appendChild(dot); el.appendChild(text);
    const marker = new maplibregl.Marker({ element: el }).setLngLat([fused.lng, fused.lat]).addTo(mapRef.current);
    estimatedMarkers.current[receiverIndex] = marker;

    // compute error (meters) vs true
    const errorMeters = haversine(rx, fused);
    // update receiver lastFix (include HPL)
    setReceivers(prev => prev.map(r => r.label === rx.label ? { ...r, lastFix: { lat: fused.lat, lng: fused.lng, err: errorMeters, hpl: estHpl } } : r));

    // push recent stats (rolling)
    setRecentErrors(prev => { const a = prev.slice(-99); a.push(errorMeters); return a; });
    setRecentHPLs(prev => {
      const a = prev.slice(-99);
      if (typeof estHpl === 'number' && !isNaN(estHpl)) a.push(estHpl);
      return a;
    });

    // log integrity event if HPL or error exceeds threshold
    // prefer a computed HPL when available, otherwise fall back to the actual error for the integrity check
    const hplCheck = (estHpl !== null && typeof estHpl === 'number') ? estHpl : errorMeters;
    if (enableIntegrityChecks && hplCheck > integrityThresholdMeters) {
      const e = { type: 'INTEGRITY_ALARM', station: refMaster.label, receiver: rx.label, errorMeters, hpl: estHpl, time: Date.now() };
      setLogEvents(prev => [...prev.slice(-400), e]);
      showToast(`INTEGRITY ALARM: ${rx.label} — check ${hplCheck.toFixed(1)} m > threshold ${integrityThresholdMeters} m (HPL ${estHpl !== null ? estHpl.toFixed(1) : 'n/a'})`, 'error', 8000);
    }

    showToast(`Estimated ${rx.label}: ${fused.lat.toFixed(6)}, ${fused.lng.toFixed(6)} -- (err ${errorMeters.toFixed(1)} m, HPL ${estHpl !== null ? estHpl.toFixed(1) : 'n/a'})`, 'success', 4000);
  }

  // reuse solver from original Loran (same algorithm)
  function solvePositionFromTDOA(pairs, initialLngLat) {
    const refLat = initialLngLat.lat;
    function latLngToXY(lat, lng) {
      const R = 6371000;
      const x = (lng * Math.PI / 180) * R * Math.cos(refLat * Math.PI / 180);
      const y = (lat * Math.PI / 180) * R;
      return { x, y };
    }
    let x0 = latLngToXY(initialLngLat.lat, initialLngLat.lng).x;
    let y0 = latLngToXY(initialLngLat.lat, initialLngLat.lng).y;
    const maxIter = 30;
    let JTJ_final = [[0,0],[0,0]];
    let r_final = [];
    for (let iter=0; iter<maxIter; iter++){
      const J = [];
      const r = [];
      for (const p of pairs) {
        const mxy = latLngToXY(p.master.lat, p.master.lng);
        const sxy = latLngToXY(p.slave.lat, p.slave.lng);
        const dM = Math.hypot(x0 - mxy.x, y0 - mxy.y);
        const dS = Math.hypot(x0 - sxy.x, y0 - sxy.y);
        const modeledDelta = (dS - dM)/C.c;
        const ri = (p.tdoaSec - modeledDelta);
        const ddx = ( (x0 - sxy.x)/dS - (x0 - mxy.x)/dM ) / C.c;
        const ddy = ( (y0 - sxy.y)/dS - (y0 - mxy.y)/dM ) / C.c;
        J.push([ddx, ddy]);
        r.push(ri);
      }
      const JTJ = [[0,0],[0,0]];
      const JTr = [0,0];
      for (let i=0;i<J.length;i++){
        const [j1,j2] = J[i];
        JTJ[0][0] += j1*j1; JTJ[0][1] += j1*j2;
        JTJ[1][0] += j2*j1; JTJ[1][1] += j2*j2;
        JTr[0] += j1 * r[i]; JTr[1] += j2 * r[i];
      }
      const det = JTJ[0][0] * JTJ[1][1] - JTJ[0][1] * JTJ[1][0];
      if (Math.abs(det) < 1e-12) break;
      const inv = [
        [JTJ[1][1] / det, -JTJ[0][1] / det],
        [-JTJ[1][0] / det, JTJ[0][0] / det]
      ];
      const dx = inv[0][0] * JTr[0] + inv[0][1] * JTr[1];
      const dy = inv[1][0] * JTr[0] + inv[1][1] * JTr[1];
      x0 += dx;
      y0 += dy;
  if (Math.hypot(dx, dy) < 1e-6) break;
  JTJ_final = JTJ;
  r_final = r.slice();
}
    const R = 6371000;
    const lat = (y0 / R) * 180 / Math.PI;
    const lng = (x0 / (R * Math.cos(refLat * Math.PI / 180))) * 180 / Math.PI;
    // estimate residual variance (seconds^2)
    const m = r_final.length;
    let sigma2 = 0;
    if (m > 2) {
      const ssum = r_final.reduce((a,b)=>a + b*b, 0);
      sigma2 = ssum / Math.max(1, m - 2);
    }
    // compute covariance in meters: cov = sigma2 * inv(JTJ_final)
    const detF = JTJ_final[0][0]*JTJ_final[1][1] - JTJ_final[0][1]*JTJ_final[1][0];
    let cov = [[0,0],[0,0]];
    if (Math.abs(detF) > 1e-12) {
      const invF = [[JTJ_final[1][1]/detF, -JTJ_final[0][1]/detF], [-JTJ_final[1][0]/detF, JTJ_final[0][0]/detF]];
      cov = [[sigma2 * invF[0][0], sigma2 * invF[0][1]],[sigma2 * invF[1][0], sigma2 * invF[1][1]]];
    }
    // compute HPL from covariance: take largest eigenvalue and scale by 3-sigma
    const trace = cov[0][0] + cov[1][1];
    const detC = cov[0][0]*cov[1][1] - cov[0][1]*cov[1][0];
    const tm = Math.sqrt(Math.max(0, (trace*trace)/4 - detC));
    const lambda1 = Math.max(0, trace/2 + tm);
    const hplMeters = 3 * Math.sqrt(Math.max(0, lambda1));

    return { lat, lng, covariance: cov, hplMeters };
  }

  // pulse simulation (reuse with small augmentation: attach DDS events and timing offsets)
  function simulatePulsesAtReceivers() {
    if (masters.length === 0 || slaves.length === 0 || receivers.length === 0) { showToast('Add masters, slaves, and receivers', 'error'); return; }
    // if a RNG seed is set in UI, apply it so simulation is repeatable
    try { if (rngSeed) setRngSeed(parseInt(rngSeed)); } catch(e) {}
    const sampleRate = 1000000;
    const pulseDuration = 0.0001;
    const totalDuration = 0.01;
    const numSamples = Math.floor(totalDuration * sampleRate);
    const results = receivers.map((r) => {
      const arrivals = [];
      // simulate periodic emissions per station using GRI and optional secondary spacing
      masters.forEach((m) => {
        const griSec = (m.griMs || 1000) / 1000;
        const phase = m.phaseSec || 0;
        // consider a few emission periods around current sim time
        const baseK = Math.floor(simTimeRef.current / griSec);
        for (let k = -1; k <= 1; k++) {
          const t_emit = (baseK + k) * griSec + phase;
          const prop = haversine({ lat: m.lat, lng: m.lng }, { lat: r.lat, lng: r.lng }) / C.c;
          const clockOffset = simulateClockTick(m.clock || { biasSec:0, driftPerSec:0 }, simTimeRef.current) || 0;
          const offsetSec = m.offsetSec || 0;
          let asfMeters = 0;
          if (m.asfMap && typeof m.asfMap === 'function') {
            try { asfMeters = m.asfMap(r.lat, r.lng) || 0; } catch(e) { asfMeters = 0; }
          } else if (m.asfMeters !== undefined) asfMeters = m.asfMeters || 0;
          let diffCorrMeters = 0;
          if (m.diffCorrections && m.diffCorrections.enabled) diffCorrMeters = m.diffCorrections.avgMeters || 0;
          const arrivalSec = t_emit + prop + clockOffset + offsetSec + (asfMeters - diffCorrMeters) / C.c;
          // detection jitter
          const detectJitterSec = gaussianNoise(detectJitterMs) / 1000;
          const arrivalWithJitter = arrivalSec + detectJitterSec;
          // include if within simulation window relative to simTimeRef
          if (arrivalWithJitter >= simTimeRef.current && arrivalWithJitter <= simTimeRef.current + totalDuration) {
            arrivals.push({ station: m.label, type: 'master', arrivalSec: arrivalWithJitter, txDbm: m.txDbm, dds: m.ddsEnabled });
            // skywave component
            if (skyEnabled) {
              const skyArrival = arrivalWithJitter + (skyDelayMs || 0) / 1000;
              arrivals.push({ station: m.label, type: 'master-sky', arrivalSec: skyArrival, txDbm: m.txDbm, txScale: skyAmpFraction });
            }
          }
        }
      });
      slaves.forEach((s) => {
        const griSec = (s.griMs || 1000) / 1000;
        const phase = s.phaseSec || 0;
        const baseK = Math.floor(simTimeRef.current / griSec);
        for (let k=-1;k<=1;k++){
          const t_emit = (baseK + k) * griSec + phase;
          const prop = haversine({ lat: s.lat, lng: s.lng }, { lat: r.lat, lng: r.lng }) / C.c;
          const clockOffset = simulateClockTick(s.clock || { biasSec:0, driftPerSec:0 }, simTimeRef.current) || 0;
          const offsetSec = s.offsetSec || 0;
          let asfMeters = 0;
          if (s.asfMap && typeof s.asfMap === 'function') {
            try { asfMeters = s.asfMap(r.lat, r.lng) || 0; } catch(e) { asfMeters = 0; }
          }
          let diffCorrMeters = 0;
          if (s.diffCorrections && s.diffCorrections.enabled) diffCorrMeters = s.diffCorrections.avgMeters || 0;
          const arrivalSec = t_emit + prop + clockOffset + offsetSec + (asfMeters - diffCorrMeters) / C.c;
          const detectJitterSec = gaussianNoise(detectJitterMs) / 1000;
          const arrivalWithJitter = arrivalSec + detectJitterSec;
          if (arrivalWithJitter >= simTimeRef.current && arrivalWithJitter <= simTimeRef.current + totalDuration) {
            arrivals.push({ station: s.label, type: 'slave', arrivalSec: arrivalWithJitter, txDbm: s.txDbm || 18 });
          }
        }
      });
      arrivals.sort((a,b)=>a.arrivalSec - b.arrivalSec);
      const waveform = new Array(numSamples).fill(0);
      arrivals.forEach((arrival) => {
        const startSample = Math.floor(Math.max(0, (arrival.arrivalSec - simTimeRef.current)) * sampleRate);
        const endSample = Math.floor(((arrival.arrivalSec - simTimeRef.current) + pulseDuration) * sampleRate);
        const amplitude = Math.pow(10, (arrival.txDbm || 20) / 20) * (arrival.txScale || 1);
        for (let i = startSample; i < endSample && i < numSamples; i++) {
          const t = (i - startSample) / sampleRate;
          const pulseShape = 0.5 * (1 + Math.cos(Math.PI * t / pulseDuration));
          waveform[i] += amplitude * pulseShape;
        }
      });
      return { receiver: r.label, arrivals, waveform, sampleRate, simStart: simTimeRef.current, totalDuration };
    });
    setSimulationResults(results);
    // broadcast DDS messages at current simulation time
    broadcastDDS(simTimeRef.current);
    showToast('Pulse & DDS simulation completed (waveforms available in sidebar).', 'success', 4000);
  }

  // basic periodic sim clock increment (for clock drift simulation)
  useEffect(() => {
    const tid = setInterval(() => {
      simTimeRef.current += 1; // increment 1s per real second (simple)
      setTimeSinceStart(simTimeRef.current);
      // optionally broadcast DDS each 5s
      if (simTimeRef.current % 5 === 0) broadcastDDS(simTimeRef.current);
    }, 1000);
    return () => clearInterval(tid);
  }, [masters, enableDDSGlobal]);

  function resetAll() {
    setMasters([]); setSlaves([]); setReceivers([]); setGridStatus(null); setSimulationResults(null); setLogEvents([]);
    // remove markers from map
    Object.values(markers.current).forEach(m => { try { m.remove(); } catch(e) {} }); markers.current = {};
    Object.values(estimatedMarkers.current).forEach(m => { try { m.remove(); } catch(e) {} }); estimatedMarkers.current = {};
    // remove LOP layers/sources and any popups
    if (mapRef.current) {
      try { mapRef.current.off('click', 'elops'); } catch(e) {}
      try { if (mapRef.current.getLayer('elops')) mapRef.current.removeLayer('elops'); } catch(e) {}
      try { if (mapRef.current.getSource('elops')) mapRef.current.removeSource('elops'); } catch(e) {}
      try { if (mapRef.current.getLayer('elop-labels')) mapRef.current.removeLayer('elop-labels'); } catch(e) {}
      try { if (mapRef.current.getSource('elop-labels')) mapRef.current.removeSource('elop-labels'); } catch(e) {}
      // remove any open popups (maplibre uses .mapboxgl-popup)
      try { document.querySelectorAll('.mapboxgl-popup').forEach(n => n.remove()); } catch(e) {}
    }
    masterCounter.current = 0; slaveCounter.current = 0; receiverCounter.current = 0;
    // reset sim time as well
    simTimeRef.current = 0;
    setTimeSinceStart(0);
    // release large grid maps
    gridMapsRef.current = null;
  }

  // Auto-calibration: compute per-master diffCorrections from simulated arrivals
  function autoCalibrate() {
    if (!simulationResults || simulationResults.length === 0) { showToast('Run simulation first (Simulate) to collect arrivals', 'error'); return; }
    const masterCorrections = {};
    masters.forEach(m => masterCorrections[m.label] = []);
    // iterate receivers
    simulationResults.forEach(res => {
      const rxLabel = res.receiver;
      const rxObj = receivers.find(r => r.label === rxLabel);
      if (!rxObj) return;
      const rxLat = rxObj.lat, rxLng = rxObj.lng;
      for (const arr of res.arrivals) {
        if (!arr || !arr.station) continue;
        const m = masters.find(mm => mm.label === arr.station);
        if (!m) continue;
        // use primary master arrivals only (ignore sky)
        if (arr.type && arr.type.includes('sky')) continue;
        const observed = arr.arrivalSec;
        const predicted = computeArrivalSecNoDiff(m, rxLat, rxLng, simTimeRef.current);
        const deltaSec = observed - predicted;
        const deltaMeters = deltaSec * C.c;
        // diffCorr to apply should be -deltaMeters (see notes)
        masterCorrections[m.label].push(-deltaMeters);
      }
    });
    // compute averages and apply
    const updates = masters.map(m => {
      const arr = masterCorrections[m.label] || [];
      if (arr.length === 0) return m;
      const avg = arr.reduce((a,b)=>a+b,0)/arr.length;
      return { ...m, diffCorrections: { enabled: true, avgMeters: avg } };
    });
    setMasters(updates);
    showToast('Auto-calibration applied to masters (avg corrections set)', 'success', 4000);
  }

  // Export simulated pulse timing (CSV)
  function exportSimTiming() {
    if (!simulationResults || simulationResults.length === 0) { showToast('No simulation results to export', 'error'); return; }
    const rows = [];
    rows.push(['receiver','station','type','arrivalSec','txDbm','txScale'].join(','));
    simulationResults.forEach(res => {
      for (const a of res.arrivals) {
        rows.push([res.receiver, a.station, a.type || '', a.arrivalSec, a.txDbm || '', a.txScale || ''].join(','));
      }
    });
    const csv = rows.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'sim_timing.csv'; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }

  // small UI for station editing (clock/diff)
  async function editMasterConfig(label) {
    const m = masters.find(x => x.label === label);
    if (!m) return;
    const newBias = await showPrompt("Clock bias in seconds (positive means station clock ahead of UTC):", String(m.clock?.biasSec || 0));
    const newDrift = await showPrompt("Clock drift (seconds per second):", String(m.clock?.driftPerSec || 0));
    const diffAvg = await showPrompt("Differential correction average (meters, positive = subtract from error):", String((m.diffCorrections && m.diffCorrections.avgMeters) || 0));
    const ddsEn = await showConfirm("Enable DDS for this master? OK=yes, Cancel=no");
    const griMs = await showPrompt("GRI (ms) for transmitted pulses (e.g. 1000):", String(m.griMs || 1000));
    const phaseSec = await showPrompt("Pulse phase offset (sec):", String(m.phaseSec || 0));
    setMasters(prev => prev.map(st => st.label === label ? { ...st, clock: { type: st.clock?.type || 'gps-disciplined', biasSec: parseFloat(newBias)||0, driftPerSec: parseFloat(newDrift)||0 }, diffCorrections: { enabled: true, avgMeters: parseFloat(diffAvg)||0 }, ddsEnabled: !!ddsEn, griMs: parseInt(griMs)||1000, phaseSec: parseFloat(phaseSec)||0 } : st));
  }

  // ASF assignment helpers (minimal UI-driven evaluator)
  async function applyAsfToMaster() {
    if (!asfTarget) { showToast('Select a master to apply ASF to', 'error'); return; }
    const code = asfText || 'return 0;';
    // Basic blacklist to avoid obvious globals and network APIs
    const blacklist = ['window','document','fetch','XMLHttpRequest','importScripts','self','postMessage','location','navigator','require'];
    for (const t of blacklist) {
      const re = new RegExp('\\b' + t + '\\b','i');
      if (re.test(code)) { showToast('ASF code contains forbidden token: ' + t, 'error'); return; }
    }

    // Validate using worker if available
    try {
      if (!asfWorkerRef.current) {
        asfWorkerRef.current = new Worker(new URL('../workers/asfWorker.js', import.meta.url), { type: 'module' });
      }
      const aw = asfWorkerRef.current;
      const testLat = masters[0]?.lat || 0;
      const testLng = masters[0]?.lng || 0;
      // Promise wrapper for worker response
      const res = new Promise((resolve, reject) => {
        const onmsg = (ev) => {
          const m = ev.data;
          if (!m) return;
          if (m.type === 'result') { aw.removeEventListener('message', onmsg); resolve(m.payload.value); }
          else if (m.type === 'error') { aw.removeEventListener('message', onmsg); reject(new Error(m.payload.message || 'ASF eval error')); }
        };
        aw.addEventListener('message', onmsg);
        aw.postMessage({ type: 'eval', payload: { code, lat: testLat, lng: testLng } });
        // timeout
        setTimeout(() => { aw.removeEventListener('message', onmsg); reject(new Error('ASF eval timeout')); }, 1200);
      });
      try {
        const val = await res;
        if (typeof val !== 'number') {
          const ok = await showConfirm('ASF worker returned non-number. Assign anyway?');
          if (!ok) return;
        }
        // create main-thread function for per-cell performance
        let fn;
        try { fn = createAsfFunctionFromText(code); } catch (e) { showToast('Error creating ASF function: '+e.message, 'error'); return; }
        setMasters(prev => prev.map(m => m.label === asfTarget ? { ...m, asfMap: fn } : m));
        showToast(`ASF assigned to ${asfTarget} (validated in worker)`, 'success', 4000);
      } catch (err) {
        showToast('ASF validation failed: ' + (err.message || String(err)), 'error', 6000);
      }
      return;
    } catch (e) {
      // fallback
      try {
        const fn = createAsfFunctionFromText(code);
        const test = fn(masters[0]?.lat || 0, masters[0]?.lng || 0);
        if (typeof test !== 'number') {
          const ok = await showConfirm('ASF function did not return a number on test call. Continue anyway?');
          if (!ok) return;
        }
        setMasters(prev => prev.map(m => m.label === asfTarget ? { ...m, asfMap: fn } : m));
        showToast(`ASF assigned to ${asfTarget}`, 'success', 4000);
      } catch (err) {
        showToast('Error creating ASF function: ' + err.message, 'error', 6000);
      }
    }
  }

  function clearAsfFromMaster() {
    if (!asfTarget) { showToast('Select a master', 'error'); return; }
    setMasters(prev => prev.map(m => m.label === asfTarget ? { ...m, asfMap: null } : m));
    showToast(`ASF cleared for ${asfTarget}`, 'success', 3000);
  }

  // Log controls: clear and export
  function clearLogs() {
    setLogEvents([]);
    showToast('Logs cleared', 'success', 2000);
  }

  function exportLogs() {
    if (!logEvents || logEvents.length === 0) { showToast('No logs to export', 'error'); return; }
    const rows = logEvents.map(ev => {
      const timestamp = ev.time ? new Date(ev.time).toISOString() : (ev.msg && ev.msg.utcMs ? new Date(ev.msg.utcMs).toISOString() : '');
      const type = ev.type || '';
      const station = ev.station || ev.msg?.from || '';
      const receiver = ev.receiver || '';
      const seq = ev.msg?.seq || '';
      const diff = (ev.msg && typeof ev.msg.diffMeters !== 'undefined') ? ev.msg.diffMeters : '';
      const err = typeof ev.errorMeters !== 'undefined' ? ev.errorMeters : '';
      const raw = JSON.stringify(ev);
      return [timestamp, type, station, receiver, seq, diff, err, raw];
    });
    const header = ['timestamp','type','station','receiver','seq','diffMeters','errorMeters','raw'];
    const csv = [header.join(','), ...rows.map(r => r.map(cell => `"${String(cell).replace(/"/g,'""')}"`).join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'eloran_logs.csv'; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }

  // UI render
  return (
    <div className="w-full h-full flex flex-col">
      {/* Toast container (fixed, left-bottom) */}
      <div className="fixed left-4 bottom-4 z-50 flex flex-col gap-2 items-start">
        {toasts.map(t => (
          <div key={t.id} className={`px-3 py-2 rounded shadow-lg max-w-xs wrap-break-word ${t.type === 'error' ? 'bg-red-600 text-white' : t.type === 'success' ? 'bg-emerald-600 text-white' : 'bg-gray-800 text-white'}`}>
            {t.msg}
          </div>
        ))}
      </div>
      <Modal open={modalState.open} type={modalState.type} title={modalState.title} message={modalState.message} value={modalInput} onChange={setModalInput} onConfirm={handleModalConfirm} onCancel={handleModalCancel} />
      <div className="px-3 py-2 bg-white border-b flex items-center gap-4">
      <h2 className="text-lg font-semibold">Eloran Simulator</h2>
        <div className="flex gap-2">
          <button onClick={() => setMode('add-master')} className={`px-2 py-1 rounded transition-transform duration-150 hover:shadow-md hover:scale-105 ${mode==='add-master' ? 'bg-sky-600 text-white' : 'bg-gray-100'}`}>Add Masters</button>
          <button onClick={() => setMode('add-slave')} className={`px-2 py-1 rounded transition-transform duration-150 hover:shadow-md hover:scale-105 ${mode==='add-slave' ? 'bg-yellow-500 text-white' : 'bg-gray-100'}`}>Add Slaves</button>
          <button onClick={() => setMode('add-receiver')} className={`px-2 py-1 rounded transition-transform duration-150 hover:shadow-md hover:scale-105 ${mode==='add-receiver' ? 'bg-green-600 text-white' : 'bg-gray-100'}`}>Add Receivers</button>
          <button onClick={() => setMode('pan')} className={`px-2 py-1 rounded transition-transform duration-150 hover:shadow-md hover:scale-105 ${mode==='pan' ? 'bg-slate-600 text-white' : 'bg-gray-100'}`}>Del. Mark</button>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <label className="px-3 py-1 rounded bg-gray-600 text-white cursor-pointer flex items-center transition-transform duration-150 hover:shadow-md hover:scale-105" title="Import stations from CSV" style={{marginBottom:0}}>
            Import CSV
            <input type="file" accept=".csv" onChange={handleCsvImport} className="hidden" />
          </label>
          <button onClick={() => computeGrid(200,200)} title="Compute TDOA grid" className="px-3 py-1 rounded bg-indigo-600 text-white transition-transform duration-150 hover:shadow-md hover:scale-105">Compute</button>
          <button onClick={() => simulatePulsesAtReceivers()} title="Run pulse + DDS simulation" className="px-3 py-1 rounded bg-emerald-600 text-white transition-transform duration-150 hover:shadow-md hover:scale-105">Simulate</button>
          <button onClick={() => estimateReceiver(0)} title="Estimate receiver position" className="px-3 py-1 rounded bg-fuchsia-600 text-white transition-transform duration-150 hover:shadow-md hover:scale-105">Estimate</button>
          <button onClick={() => resetAll()} title="Reset everything" className="px-3 py-1 rounded bg-red-600 text-white transition-transform duration-150 hover:shadow-md hover:scale-105">Reset</button>
        </div>
      </div>

      <div className="flex h-[calc(100vh-72px)]">
        <div className="w-3/4" ref={mapContainer} />

        <aside className="w-1/4 bg-white border-l p-3 overflow-auto">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Eloran Controls & Status</h3>
            <div className="text-xs text-gray-600 flex items-center gap-2">
              <div title="Active RNG seed">Seed: <span className="font-medium">{rngSeed || '—'}</span></div>
              <button onClick={()=>{ try { navigator.clipboard.writeText(String(rngSeed || '')); showToast('Seed copied', 'success', 1500); } catch(e){ showToast('Clipboard not available','error',1500); } }} className="text-xs px-2 py-0.5 rounded bg-gray-100">Copy</button>
            </div>
          </div>
          <div className="mt-2 text-xs">
              <div>Sim time: {timeSinceStart}s</div>
              <label className="block mt-2"><input type="checkbox" checked={enableDDSGlobal} onChange={(e)=>setEnableDDSGlobal(e.target.checked)} /> Global DDS enabled</label>
              <label className="block mt-1"><input type="checkbox" checked={enableIntegrityChecks} onChange={(e)=>setEnableIntegrityChecks(e.target.checked)} /> Integrity checks</label>
              <div className="mt-1">Integrity threshold (m): <input type="number" value={integrityThresholdMeters} onChange={(e)=>setIntegrityThresholdMeters(parseFloat(e.target.value)||0)} style={{width:80}} /></div>
              <div className="mt-1">Contour simplify (m): <input type="number" min="0" step="1" value={contourSimplifyToleranceMeters} onChange={(e)=>setContourSimplifyToleranceMeters(parseFloat(e.target.value)||0)} style={{width:80}} /></div>
            </div>

          <div className="mt-4">
            <div className="flex items-center justify-between">
              <h4 className="font-medium">Masters ({masters.length})</h4>
              <div className="flex gap-2">
                <button onClick={() => { setMasters(prev => prev.map(m => ({ ...m, diffCorrections: { enabled: false, avgMeters: 0 } }))); showToast('Master diffCorrections reset', 'success'); }} className="text-xs px-2 py-0.5 rounded bg-gray-100" title="Clear per-master diff corrections">Reset DiffCorr</button>
              </div>
            </div>
            <div className="text-xs mt-2 space-y-1">
              {masters.map((m,i) => (
                <div key={m.label} className="border rounded p-1">
                  <div className="flex justify-between items-center">
                    <div><strong>{m.label}</strong> ({m.lat.toFixed(4)},{m.lng.toFixed(4)})</div>
                    <div className="flex gap-1">
                      <button onClick={()=>editMasterConfig(m.label)} className="text-xs px-2 py-0.5 rounded bg-gray-100 transition-transform duration-150 hover:shadow-md hover:scale-105">Edit</button>
                    </div>
                  </div>
                  <div className="text-xs mt-1">Clock: {m.clock?.type || 'gps-disciplined'} (bias {m.clock?.biasSec || 0}s)</div>
                  <div className="text-xs">DDS: {m.ddsEnabled ? 'ON' : 'OFF'} — DiffCorr: {m.diffCorrections?.avgMeters ?? 0} m</div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4">
            <h4 className="font-medium">ASF Assignment</h4>
            <div className="text-xs mt-2">
              <div className="flex gap-2 items-center">
                <select value={asfTarget} onChange={(e)=>setAsfTarget(e.target.value)} className="text-xs">
                  <option value="">-- select master --</option>
                  {masters.map(m => (<option key={m.label} value={m.label}>{m.label}</option>))}
                </select>
                <button onClick={applyAsfToMaster} className="text-xs px-2 py-0.5 rounded bg-indigo-100 transition-transform duration-150 hover:shadow-md hover:scale-105">Apply</button>
                <button onClick={clearAsfFromMaster} className="text-xs px-2 py-0.5 rounded bg-gray-100 transition-transform duration-150 hover:shadow-md hover:scale-105">Clear</button>
              </div>
              <div className="mt-2">
                <label className="block text-xs">ASF JS (function body). Example: `return 100*Math.sin(lat);` (meters). Args: `lat, lng`</label>
                <textarea value={asfText} onChange={(e)=>setAsfText(e.target.value)} className="w-full text-xs" rows={4} />
              </div>
            </div>
          </div>

          <div className="mt-2">
            <label className="block text-xs">Contour units:</label>
            <select value={contourUnit} onChange={(e)=>{ setContourUnit(e.target.value); if (gridStatus?.contours) drawLOPs(gridStatus.contours); }} className="text-xs">
              <option value="meters">Meters</option>
              <option value="seconds">Seconds</option>
            </select>
          </div>

          <div className="mt-4">
            <h4 className="font-medium">Receivers ({receivers.length})</h4>
            <div className="text-xs mt-2 space-y-1">
              {receivers.map((r, idx) => (
                <div key={r.label} className="border rounded p-1">
                  <div className="flex justify-between items-center">
                    <div>[{r.label}] {r.lat.toFixed(4)},{r.lng.toFixed(4)}</div>
                    <div className="flex gap-1">
                      <button onClick={()=>estimateReceiver(idx)} className="text-xs px-2 py-0.5 rounded bg-indigo-100 transition-transform duration-150 hover:shadow-md hover:scale-105">Est</button>
                      <select value={r.fuseMode} onChange={(e)=> setReceivers(prev => prev.map(x => x.label===r.label ? {...x, fuseMode: e.target.value} : x)) } className="text-xs">
                        <option value="Eloran">Eloran</option>
                        <option value="GNSS">GNSS</option>
                        <option value="fusion">Fusion</option>
                      </select>
                    </div>
                  </div>
                  <div className="text-xs mt-1">Last err: {r.lastFix ? `${r.lastFix.err.toFixed(1)} m` : '-'}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-3 space-y-3">
            <div className="p-3 bg-white/80 rounded-lg shadow-sm border">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold">Estimator</h4>
                <div className="text-xs text-gray-500">HPL-aware</div>
              </div>
              <div className="mt-2 grid grid-cols-1 gap-2">
                <div className="flex items-center gap-2">
                  <label className="text-sm">Mode:</label>
                  <div className="flex items-center gap-2">
                    <label className="inline-flex items-center text-xs"><input type="radio" name="estMode" value="controlled" checked={estimatorMode==='controlled'} onChange={(e)=>setEstimatorMode(e.target.value)} className="mr-1"/>Controlled</label>
                    <label className="inline-flex items-center text-xs"><input type="radio" name="estMode" value="random" checked={estimatorMode==='random'} onChange={(e)=>setEstimatorMode(e.target.value)} className="mr-1"/>Random</label>
                    <label className="inline-flex items-center text-xs"><input type="radio" name="estMode" value="none" checked={estimatorMode==='none'} onChange={(e)=>setEstimatorMode(e.target.value)} className="mr-1"/>None</label>
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <div>Noise σ (meters)</div>
                    <div className="font-medium">{estNoiseStdMeters.toFixed ? estNoiseStdMeters.toFixed(1) : estNoiseStdMeters} m</div>
                  </div>
                  <input type="number" min="0" max="200" step="0.5" value={estNoiseStdMeters} onChange={(e)=>setEstNoiseStdMeters(parseFloat(e.target.value)||0)} className="w-full mt-1 px-2 py-1 border rounded text-sm" />
                </div>
                <div className="text-xs text-gray-500">Controlled: fixed Gaussian spread • Random: variable amplitude • None: ideal (no added noise)</div>
              </div>
            </div>

            <div className="p-3 bg-white/80 rounded-lg shadow-sm border">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold">Random Seed</h4>
                <div className="text-xs text-gray-500">Reproducibility</div>
              </div>
              <div className="mt-2 flex flex-col gap-2">
                <input type="text" value={rngSeed} onChange={(e)=>setRngSeedState(e.target.value)} className="px-2 py-1 border rounded text-sm flex-1 min-w-0" />
                <button onClick={()=>{ const s = parseInt(rngSeed); if (Number.isNaN(s)) { const seed = Math.floor(Date.now()%4294967296); setRngSeed(seed); try { localStorage.setItem('eloran_rng_seed', String(seed)); setRngSeedState(String(seed)); } catch(e){} showToast('Applied time-based seed: '+seed,'success',3000); } else { setRngSeed(s); try { localStorage.setItem('eloran_rng_seed', String(s)); } catch(e){} showToast('Seed applied: '+s,'success',3000); } }} className="px-3 py-1 text-sm bg-gray-100 rounded transition-transform duration-150 hover:shadow-md hover:scale-105 shrink-0" title="Apply seed">Apply</button>
                <button onClick={()=>{ const seed = Math.floor(_rng()*4294967296); setRngSeed(seed); try { localStorage.setItem('eloran_rng_seed', String(seed)); setRngSeedState(String(seed)); } catch(e){} showToast('Random seed set: '+seed,'success',3000); }} className="px-3 py-1 text-sm bg-gray-50 rounded transition-transform duration-150 hover:shadow-md hover:scale-105 shrink-0" title="Generate random seed">Randomize</button>
                <button onClick={()=>{ try { navigator.clipboard.writeText(String(rngSeed)); showToast('Copied seed to clipboard','success',2000); } catch(e){ showToast('Clipboard not available','error',2000); } }} className="px-2 py-1 text-sm bg-white border rounded transition-transform duration-150 hover:shadow-md hover:scale-105 shrink-0" title="Copy seed">Copy</button>
              </div>
              <div className="text-xs text-gray-500 mt-2">Use seed to reproduce stochastic simulator runs and estimator draws.</div>
            </div>

            <div className="p-3 bg-white/80 rounded-lg shadow-sm border">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold">Pulse & Channel</h4>
                <div className="text-xs text-gray-500">Jitter · Skywave</div>
              </div>
              <div className="mt-2 flex flex-col gap-2">
                <div>
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <div>Detection jitter (ms)</div>
                    <div className="font-medium">{detectJitterMs} ms</div>
                  </div>
                  <input type="number" min="0" max="20" step="0.1" value={detectJitterMs} onChange={(e)=>setDetectJitterMs(parseFloat(e.target.value)||0)} className="w-full mt-1 px-2 py-1 border rounded text-sm" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="inline-flex items-center text-sm"><input type="checkbox" checked={skyEnabled} onChange={(e)=>setSkyEnabled(e.target.checked)} className="mr-2"/>Enable skywave</label>
                </div>
                {skyEnabled && (
                  <div className="grid grid-cols-1 gap-2">
                    <div className="flex items-center justify-between text-xs text-gray-600"> <div>Sky delay (ms)</div><div className="font-medium">{skyDelayMs} ms</div></div>
                    <input type="number" min="0" max="50" step="0.5" value={skyDelayMs} onChange={(e)=>setSkyDelayMs(parseFloat(e.target.value)||0)} className="w-full px-2 py-1 border rounded text-sm" />
                    <div className="flex items-center justify-between text-xs text-gray-600"> <div>Sky amplitude</div><div className="font-medium">{(skyAmpFraction*100).toFixed(0)}%</div></div>
                    <input type="number" min="0" max="1" step="0.01" value={skyAmpFraction} onChange={(e)=>setSkyAmpFraction(parseFloat(e.target.value)||0)} className="w-full px-2 py-1 border rounded text-sm" />
                  </div>
                )}
                <div className="text-xs text-gray-500">Skywave simulates a delayed weaker arrival; reduce for cleaner TDOA.</div>
              </div>
            </div>

            <div className="flex gap-2">
              <button onClick={autoCalibrate} className="flex-1 px-3 py-2 bg-indigo-600 text-white rounded shadow-sm transition-transform duration-150 hover:shadow-md hover:scale-105">Auto-calibrate</button>
              <button onClick={exportSimTiming} className="px-3 py-2 bg-emerald-600 text-white rounded shadow-sm transition-transform duration-150 hover:shadow-md hover:scale-105">Export Timing CSV</button>
            </div>
            <div className="text-xs text-gray-500">Auto-calibrate sets per-master differential corrections using recent simulation arrivals.</div>
          </div>

          <div className="mt-4">
            <div className="flex items-center justify-between">
              <h4 className="font-medium">Logs (latest)</h4>
              <div className="flex gap-2">
                <button onClick={clearLogs} className="text-xs px-2 py-0.5 rounded bg-red-100 transition-transform duration-150 hover:shadow-md hover:scale-105">Clear</button>
                <button onClick={exportLogs} className="text-xs px-2 py-0.5 rounded bg-green-100 transition-transform duration-150 hover:shadow-md hover:scale-105">Export</button>
              </div>
            </div>
            <div className="text-xs mt-2 max-h-48 overflow-auto">
              {logEvents.slice().reverse().map((ev, i) => (
                <div key={i} className="border-b py-1">
                  <div className="font-semibold text-xs">{ev.type || ev.msg?.seq ? 'DDS' : 'EVENT' } — {ev.station || ev.msg?.from || ev.type}</div>
                  <div className="text-xs">{ev.msg ? `seq ${ev.msg.seq} diff ${ev.msg.diffMeters} m` : (ev.errorMeters ? `err ${ev.errorMeters.toFixed(1)} m` : JSON.stringify(ev).slice(0,80))}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Waveforms viewer: shows combined receiver waveform and arrival markers */}
          <div className="mt-4">
            <Waveforms simulationResults={simulationResults} masters={masters} slaves={slaves} receivers={receivers} />
          </div>

          <div className="mt-4 text-xs">
            <h4 className="font-medium">Tips</h4>
            <ol className="list-decimal ml-4">
              <li>Set master clock bias/drift to model UTC-traceable sources (GPS-disciplined vs local oscillator).</li>
              <li>Use DiffCorr avgMeters to simulate a local reference network improving position.</li>
              <li>Provide asfMap as a JS function on master (via console) to emulate dynamic ASF variations.</li>
            </ol>
          </div>

          <div className="mt-4">
            <h4 className="font-medium">Recent Stats</h4>
            <div className="text-xs mt-2">
              <div>Errors (m):</div>
              <div style={{display:'flex',alignItems:'flex-end',gap:2,height:40,overflow:'hidden'}}>
                {recentErrors.slice(-40).map((v,i)=> {
                  const h = Math.min(40, (v / (integrityThresholdMeters*2)) * 40 + 1);
                  return <div key={i} title={v.toFixed(1)} style={{width:6,height:h,background:'#f97316'}} />;
                })}
              </div>
              <div className="mt-2">HPLs (m):</div>
              <div style={{display:'flex',alignItems:'flex-end',gap:2,height:40,overflow:'hidden'}}>
                {recentHPLs.slice(-40).map((v,i)=> {
                  const h = Math.min(40, (v / (integrityThresholdMeters*2)) * 40 + 1);
                  return <div key={i} title={v.toFixed(1)} style={{width:6,height:h,background:'#06b6d4'}} />;
                })}
              </div>
            </div>
          </div>

          {/* Import CSV removed from sidebar */}
        </aside>
      </div>
    </div>
  );
}
