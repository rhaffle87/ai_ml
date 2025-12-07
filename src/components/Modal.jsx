import React from 'react';

export default function Modal({ open, type='info', title='', message='', defaultValue='', value, onChange, onConfirm, onCancel }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-60 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded shadow-lg max-w-lg w-full p-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">{title}</h3>
          <button onClick={onCancel} className="text-gray-500">✕</button>
        </div>
        <div className="mt-2 text-sm text-gray-700">{message}</div>
        {type === 'prompt' && (
          <div className="mt-3">
            <input autoFocus value={value} onChange={(e)=>onChange(e.target.value)} className="w-full px-2 py-1 border rounded" />
          </div>
        )}
        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onCancel} className="px-3 py-1 bg-gray-100 rounded">Cancel</button>
          <button onClick={() => onConfirm(value)} className="px-3 py-1 bg-sky-600 text-white rounded">OK</button>
        </div>
      </div>
    </div>
  );
}
