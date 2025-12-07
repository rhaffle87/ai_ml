import React, { useMemo, useState } from 'react';
import './Waveforms.css';

const C = { c: 299792458 };
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

// Simple SVG waveform + arrival markers viewer
export default function Waveforms({ simulationResults, masters = [], slaves = [], receivers = [] }) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [overlayMap, setOverlayMap] = useState({});

  const receiverNames = useMemo(() => (simulationResults || []).map(r => r.receiver), [simulationResults]);
  // safe accessors: allow rendering even when simulationResults is null
  const result = (simulationResults && simulationResults.length) ? simulationResults[Math.min(selectedIdx, simulationResults.length - 1)] : null;
  const { waveform = [], sampleRate = 1000000, arrivals = [] } = result || {};

  // downsample waveform to <= 1000 points for SVG — keep as a hook so ordering is stable
  const { pts, stations, stationRows, griLines, totalSec, simStart, griMs } = useMemo(() => {
    // prefer authoritative simStart/totalDuration from simulator results when available
    const simStart = result && result.simStart !== undefined ? result.simStart : (() => {
      const firstNonZero = waveform.findIndex(v => Math.abs(v) > 1e-12);
      const minArrival = arrivals.length ? Math.min(...arrivals.map(a => a.arrivalSec)) : 0;
      return (firstNonZero >= 0) ? (minArrival - (firstNonZero / sampleRate)) : (minArrival - 0);
    })();

    // prepare stacked per-station waveforms
    const stations = [...masters.map(m => ({ ...m, role: 'master' })), ...slaves.map(s => ({ ...s, role: 'slave' }))];
    const stationRows = stations.map(st => {
      const myArr = arrivals.filter(a => a.station === st.label);
      const samples = waveform.length || Math.max(1, Math.floor(0.01 * sampleRate));
      const row = new Float32Array(samples);
      const pulseDuration = 0.0001; // as used in simulator
      myArr.forEach(a => {
        // compute approximate tx time using geo delay
        const rxObj = receivers.find(r => r.label === (result && result.receiver));
        if (!rxObj) return;
        const prop = haversine(st, rxObj) / C.c;
        const txSec = a.arrivalSec - prop;
        const start = Math.max(0, Math.floor((a.arrivalSec - simStart) * sampleRate));
        const end = Math.min(samples, Math.floor(((a.arrivalSec - simStart) + pulseDuration) * sampleRate));
        const amplitude = Math.pow(10, (a.txDbm || 20) / 20) * (a.txScale || 1);
        for (let i = start; i < end; i++) {
          const t = (i - start) / sampleRate;
          const pulseShape = 0.5 * (1 + Math.cos(Math.PI * t / pulseDuration));
          row[i] += amplitude * pulseShape;
        }
      });
      return { station: st, data: row };
    });

    // choose a GRI for gridlines: use smallest GRI among masters+slaves or default 1000ms
    const allStations = [...masters, ...slaves];
    let griMs = 1000;
    if (allStations && allStations.length) {
      const g = allStations.map(s => Number(s.griMs || 1000)).filter(n => !Number.isNaN(n) && n > 0);
      if (g.length) griMs = Math.min(...g);
    }

    // compute vertical grid positions for GRI in the SVG coordinate space (1000 wide)
    const totalSec = result && result.totalDuration ? result.totalDuration : (waveform.length / sampleRate || 0.01);
    const griSec = griMs / 1000;
    const griLines = [];
    for (let t = 0; t < totalSec + griSec; t += griSec) {
      const x = (t / totalSec) * 1000;
      griLines.push(x);
    }

    // downsample combined waveform
    const maxPts = 1000;
    const step = Math.max(1, Math.floor((waveform.length || 1) / maxPts));
    let pts = [];
    let maxV = 1e-12;
    for (let i = 0; i < waveform.length; i += step) if (Math.abs(waveform[i]) > maxV) maxV = Math.abs(waveform[i]);
    for (let i = 0; i < waveform.length; i += step) {
      const x = (i / Math.max(1, waveform.length)) * 1000;
      const y = 25 - ((waveform[i] || 0) / (maxV || 1)) * 20;
      pts.push(`${x},${y}`);
    }

    return { pts: pts.join(' '), stations, stationRows, griLines, totalSec, simStart, griMs };
  }, [simulationResults, selectedIdx, overlayMap, masters, slaves, receivers]);

  // initialize overlayMap for stations (default visible)
  React.useEffect(() => {
    const init = {};
    stations.forEach(s => { init[s.label] = overlayMap[s.label] !== undefined ? overlayMap[s.label] : true; });
    // only set if different to avoid re-renders
    const keysEqual = Object.keys(init).length === Object.keys(overlayMap).length && Object.keys(init).every(k => overlayMap[k] === init[k]);
    if (!keysEqual) setOverlayMap(init);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stations.map(s => s.label).join(',')]);

  if (!simulationResults || simulationResults.length === 0) return (
    <div className="wf-panel">
      <div className="wf-title">Waveforms</div>
      <div className="wf-empty">Run simulation (Simulate) to show waveforms</div>
    </div>
  );

  return (
    <div className="wf-panel">
      <div className="wf-header">
        <div className="wf-title">Waveforms</div>
        <div>
          <select value={selectedIdx} onChange={(e) => setSelectedIdx(parseInt(e.target.value))} className="wf-select">
            {receiverNames.map((r,i)=> <option key={i} value={i}>{r}</option>)}
          </select>
        </div>
      </div>
      <div className="wf-legend">
        <span className="wf-legend-item"><span className="wf-dot master"/>Master</span>
        <span className="wf-legend-item"><span className="wf-dot slave"/>Slave</span>
      </div>
      <div className="wf-overlay-controls">
        {stations.map((s, i) => (
          <label key={s.label} className="wf-overlay-item">
            <input type="checkbox" checked={!!overlayMap[s.label]} onChange={(e) => setOverlayMap(prev => ({ ...prev, [s.label]: e.target.checked }))} />
            <span className="wf-overlay-swatch" style={{ background: s.role === 'master' ? '#1e90ff' : '#f59e0b' }} />{s.label}
          </label>
        ))}
      </div>
      <div className="wf-plot">
        <svg viewBox="0 0 1000 50" preserveAspectRatio="none" width="100%" height="100">
          {/* GRI gridlines */}
          {griLines.map((x, i) => (
            <line key={'g'+i} x1={x} x2={x} y1={0} y2={50} stroke="#ddd" strokeWidth={0.5} strokeDasharray="2 2" />
          ))}
          {/* waveform */}
          <polyline points={pts} fill="none" stroke="#2563eb" strokeWidth={1} />
          {/* overlay each station's contribution (faint) */}
          {stationRows.map((row, si) => {
            if (!overlayMap[row.station.label]) return null;
            const data = row.data || new Float32Array(0);
            const maxPts = 1000;
            const step = Math.max(1, Math.floor((data.length || 1) / maxPts));
            let ptsArr = [];
            let maxV = 1e-12;
            for (let i = 0; i < data.length; i += step) if (Math.abs(data[i]) > maxV) maxV = Math.abs(data[i]);
            for (let i = 0; i < data.length; i += step) {
              const x = (i / Math.max(1, data.length)) * 1000;
              const y = 50 - ((data[i] || 0) / (maxV || 1)) * 40; // same vertical scale as combined
              ptsArr.push(`${x},${y}`);
            }
            const color = row.station.role === 'master' ? '#1e90ff' : '#f59e0b';
            return (
              <polyline key={'s'+si} points={ptsArr.length ? ptsArr.join(' ') : '0,25 1000,25'} fill="none" stroke={color} strokeWidth={1} opacity={0.35} />
            );
          })}

          {/* global TX markers (triangles) for all stations' arrivals */}
          {stationRows.map((row, si) => {
            if (!overlayMap[row.station.label]) return null;
            const myArr = arrivals.filter(a => a.station === row.station.label);
            const color = row.station.role === 'master' ? '#1e90ff' : '#f59e0b';
            return myArr.map((a, i) => {
              const rxObj = receivers.find(r => r.label === (result && result.receiver));
              if (!rxObj) return null;
              const prop = haversine(row.station, rxObj) / C.c;
              const tx = a.arrivalSec - prop;
              const txX = Math.max(0, Math.min(1000, ((tx - simStart) / totalSec) * 1000));
              return (
                <polygon key={`tx-${si}-${i}`} points={`${txX},4 ${txX-4},8 ${txX+4},8`} fill={color} opacity={0.9} />
              );
            });
          })}
          {/* arrival markers */}
          {arrivals.map((a, i) => {
            const tRel = a.arrivalSec - simStart;
            const x = Math.max(0, Math.min(1000, (tRel / totalSec) * 1000));
            const color = a.type && a.type.startsWith('master') ? '#1e90ff' : '#f59e0b';
            return (
              <g key={i}>
                <line x1={x} x2={x} y1={0} y2={50} stroke={color} strokeWidth={1} opacity={0.9} />
                <text x={x+2} y={10} fontSize={8} fill={color}>{a.station}</text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* stacked per-station rows (separate from plot to avoid layout overlap) */}
      <div className="wf-rows">
        {stationRows.map((row, idx) => {
          const data = row.data || new Float32Array(0);
          // downsample to 400 points
          const maxPts = 400;
          const step = Math.max(1, Math.floor((data.length || 1) / maxPts));
          let pts = [];
          let maxV = 1e-12;
          for (let i = 0; i < data.length; i += step) if (Math.abs(data[i]) > maxV) maxV = Math.abs(data[i]);
          for (let i = 0; i < data.length; i += step) {
            const x = (i / Math.max(1, data.length)) * 1000;
            const y = 12 - ((data[i] || 0) / (maxV || 1)) * 10;
            pts.push(`${x},${y}`);
          }
          const station = row.station;
          // compute tx and rx markers for this station's arrivals
          const myArr = arrivals.filter(a => a.station === station.label);
          return (
            <div key={idx} className="wf-row">
              <div className="wf-row-label">{station.label} <span className="wf-amp">{(maxV).toFixed(2)}</span></div>
              <svg viewBox="0 0 1000 24" preserveAspectRatio="none" width="100%" height="24">
                <polyline points={pts.length ? pts.join(' ') : '0,12 1000,12'} fill="none" stroke={station.role === 'master' ? '#1e90ff' : '#f59e0b'} strokeWidth={1} />
                {/* draw tx and arrival markers */}
                {myArr.map((a, i) => {
                  const rxObj = receivers.find(r => r.label === (result && result.receiver));
                  if (!rxObj) return null;
                  const prop = haversine(station, rxObj) / C.c;
                  const tx = a.arrivalSec - prop;
                  const txX = Math.max(0, Math.min(1000, ((tx - simStart) / totalSec) * 1000));
                  const rxX = Math.max(0, Math.min(1000, ((a.arrivalSec - simStart) / totalSec) * 1000));
                  return (
                    <g key={i}>
                      <polygon points={`${txX},20 ${txX-4},24 ${txX+4},24`} fill="#22c55e" opacity={0.9} />
                      <line x1={rxX} x2={rxX} y1={0} y2={24} stroke="#444" strokeWidth={0.5} opacity={0.7} />
                    </g>
                  );
                })}
              </svg>
            </div>
          );
        })}
      </div>
      <div className="wf-info text-xs">Duration: {(waveform.length / sampleRate * 1000).toFixed(2)} ms • SampleRate: {sampleRate} Hz • GRI: {griMs} ms</div>
    </div>
  );
}
