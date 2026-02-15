import React from 'react';
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import { useColorMode } from '@docusaurus/theme-common';

const BAR_WIDTH = 20;

// ---- Theme ----

function useTheme() {
  const { colorMode } = useColorMode();
  const dark = colorMode === 'dark';
  return {
    dark,
    primary: dark ? '#60A5FA' : '#2563EB',
    primaryMuted: dark ? '#3B82F6' : '#93C5FD',
    baseline: dark ? '#6B7280' : '#9CA3AF',
    baselineFill: dark ? '#4B5563' : '#D1D5DB',
    grid: dark ? '#1F2937' : '#F3F4F6',
    axis: dark ? '#6B7280' : '#9CA3AF',
    text: dark ? '#D1D5DB' : '#4B5563',
    textMuted: dark ? '#9CA3AF' : '#6B7280',
    tooltipBg: dark ? '#111827' : '#FFFFFF',
    tooltipBorder: dark ? '#374151' : '#E5E7EB',
    bg: 'transparent',
  };
}

// ---- Tooltip ----

function ChartTooltip({ active, payload, label, suffix = '' }) {
  const t = useTheme();
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: t.tooltipBg,
      border: `1px solid ${t.tooltipBorder}`,
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 13,
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    }}>
      <p style={{ margin: 0, fontWeight: 600, color: t.text, fontSize: 12 }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ margin: '3px 0 0', color: entry.color, fontSize: 13 }}>
          {entry.name}: <strong>{typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}{suffix}</strong>
        </p>
      ))}
    </div>
  );
}

// ---- Custom legend ----

function SimpleLegend({ primary, primaryLabel, baselineLabel }) {
  const t = useTheme();
  return (
    <div style={{ display: 'flex', gap: 20, justifyContent: 'center', marginBottom: 4, fontSize: 13 }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ width: 24, height: 3, background: primary || t.primary, borderRadius: 2, display: 'inline-block' }} />
        <span style={{ color: t.text, fontWeight: 600 }}>{primaryLabel || 'ztensor'}</span>
      </span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ width: 24, height: 2, background: t.baseline, borderRadius: 2, display: 'inline-block' }} />
        <span style={{ color: t.textMuted }}>{baselineLabel || 'baselines'}</span>
      </span>
    </div>
  );
}

// ---- Line chart ----

const BASELINE_FORMATS = ['safetensors', 'pickle', 'hdf5', 'gguf'];
const BASELINE_LABELS = { safetensors: 'safetensors', pickle: 'pickle', hdf5: 'hdf5', gguf: 'gguf' };

export function ThroughputChart({ data, formats, height = 320, yLabel = 'Read Throughput (GB/s)', showZstd = false }) {
  const t = useTheme();
  const baselines = formats.filter(f => BASELINE_FORMATS.includes(f));

  return (
    <div style={{ margin: '16px 0' }}>
      <SimpleLegend />
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 20, left: 4 }}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis
            dataKey="size"
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={{ stroke: t.axis }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 12, fill: t.textMuted, fontSize: 12 }}
            domain={[0, 'auto']}
            width={50}
          />
          <Tooltip content={<ChartTooltip suffix=" GB/s" />} />
          {baselines.map((fmt) => (
            <Line
              key={fmt}
              type="monotone"
              dataKey={fmt}
              name={BASELINE_LABELS[fmt] || fmt}
              stroke={t.baseline}
              strokeWidth={1.5}
              dot={{ r: 3, fill: t.baseline, strokeWidth: 0 }}
              activeDot={{ r: 5, fill: t.baseline }}
            />
          ))}
          {showZstd && (
            <Line
              type="monotone"
              dataKey="ztensor_zstd"
              name="ztensor (zstd-3)"
              stroke={t.primaryMuted}
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={{ r: 3, fill: t.primaryMuted, strokeWidth: 0 }}
              activeDot={{ r: 5, fill: t.primaryMuted }}
            />
          )}
          <Line
            type="monotone"
            dataKey="ztensor"
            name="ztensor"
            stroke={t.primary}
            strokeWidth={3}
            dot={{ r: 5, fill: t.primary, strokeWidth: 0 }}
            activeDot={{ r: 7, fill: t.primary }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---- Bar charts ----

export function SelectiveBarChart({ data, height = 280 }) {
  const t = useTheme();
  return (
    <div style={{ margin: '16px 0' }}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} barSize={BAR_WIDTH}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis
            dataKey="name"
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={{ stroke: t.axis }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            label={{ value: 'Latency (seconds)', angle: -90, position: 'insideLeft', offset: 12, fill: t.textMuted, fontSize: 12 }}
            domain={[0, 'auto']}
            width={50}
          />
          <Tooltip content={<ChartTooltip suffix="s" />} />
          <Bar dataKey="value" name="Latency" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.id === 'ztensor' ? t.primary : t.baselineFill}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ZeroCopyBarChart({ data, height = 280 }) {
  const t = useTheme();
  return (
    <div style={{ margin: '16px 0' }}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} barSize={BAR_WIDTH}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis
            dataKey="name"
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={{ stroke: t.axis }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            label={{ value: 'Read Throughput (GB/s)', angle: -90, position: 'insideLeft', offset: 12, fill: t.textMuted, fontSize: 12 }}
            domain={[0, 'auto']}
            width={50}
          />
          <Tooltip content={<ChartTooltip suffix=" GB/s" />} />
          <Legend
            formatter={(value) => <span style={{ color: t.text, fontSize: 12 }}>{value}</span>}
          />
          <Bar dataKey="copy=False" name="copy=False (default)" fill={t.primary} radius={[4, 4, 0, 0]} />
          <Bar dataKey="copy=True" name="copy=True" fill={t.baselineFill} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CompressionBarChart({ data, height = 240 }) {
  const t = useTheme();
  return (
    <div style={{ margin: '16px 0' }}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} barSize={BAR_WIDTH}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis
            dataKey="name"
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={{ stroke: t.axis }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            label={{ value: 'File Size (MB)', angle: -90, position: 'insideLeft', offset: 12, fill: t.textMuted, fontSize: 12 }}
            domain={[0, 2100]}
            width={50}
          />
          <Tooltip content={<ChartTooltip suffix=" MB" />} />
          <ReferenceLine y={2048} stroke={t.axis} strokeDasharray="4 4" label={{ value: '2048 MB (original)', fill: t.textMuted, fontSize: 11, position: 'right' }} />
          <Bar dataKey="size" name="zstd-3 File Size" fill={t.primary} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---- Cross-format bar chart ----

export function CrossFormatBarChart({ data, height = 320 }) {
  const t = useTheme();
  return (
    <div style={{ margin: '16px 0' }}>
      <div style={{ display: 'flex', gap: 20, justifyContent: 'center', marginBottom: 4, fontSize: 13 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 14, background: t.primary, borderRadius: 3, display: 'inline-block' }} />
          <span style={{ color: t.text, fontWeight: 600 }}>ztensor</span>
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 14, background: t.baselineFill, borderRadius: 3, display: 'inline-block' }} />
          <span style={{ color: t.textMuted }}>reference impl.</span>
        </span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} barSize={BAR_WIDTH}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis
            dataKey="name"
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={{ stroke: t.axis }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            label={{ value: 'Read Throughput (GB/s)', angle: -90, position: 'insideLeft', offset: 12, fill: t.textMuted, fontSize: 12 }}
            domain={[0, 'auto']}
            width={50}
          />
          <Tooltip content={<ChartTooltip suffix=" GB/s" />} />
          <Bar dataKey="ztensor" name="ztensor" fill={t.primary} radius={[4, 4, 0, 0]} />
          <Bar dataKey="native" name="Reference impl." fill={t.baselineFill} radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.native != null ? t.baselineFill : 'transparent'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---- Distribution comparison (grouped bar) ----

const FORMAT_COLORS = {
  ztensor: { light: '#2563EB', dark: '#60A5FA' },
  gguf: { light: '#9333EA', dark: '#C084FC' },
  safetensors: { light: '#DC2626', dark: '#F87171' },
  pickle: { light: '#16A34A', dark: '#4ADE80' },
  hdf5: { light: '#D97706', dark: '#FBBF24' },
};

function DistributionBarChart({ data, yLabel = 'Throughput (GB/s)', yMax = 'auto', height = 340 }) {
  const t = useTheme();
  const blue = {
    large: t.dark ? '#3B82F6' : '#2563EB',
    mixed: t.dark ? '#60A5FA' : '#3B82F6',
    small: t.dark ? '#93C5FD' : '#60A5FA',
  };
  const gray = {
    large: t.dark ? '#6B7280' : '#9CA3AF',
    mixed: t.dark ? '#9CA3AF' : '#C0C5CE',
    small: t.dark ? '#C0C5CE' : '#D1D5DB',
  };
  return (
    <div style={{ margin: '16px 0' }}>
      <div style={{ display: 'flex', gap: 20, justifyContent: 'center', marginBottom: 4, fontSize: 13 }}>
        {Object.entries({ large: 'Large', mixed: 'Mixed', small: 'Small' }).map(([key, label]) => (
          <span key={key} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 14, height: 14, background: blue[key], borderRadius: 3, display: 'inline-block' }} />
            <span style={{ color: t.text, fontSize: 12 }}>{label}</span>
          </span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} barSize={BAR_WIDTH}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis
            dataKey="name"
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={{ stroke: t.axis }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 12, fill: t.textMuted, fontSize: 12 }}
            domain={[0, yMax]}
            width={50}
          />
          <Tooltip content={<ChartTooltip suffix=" GB/s" />} />
          {['large', 'mixed', 'small'].map(key => (
            <Bar key={key} dataKey={key} name={key.charAt(0).toUpperCase() + key.slice(1)} radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.name === 'ztensor' ? blue[key] : gray[key]} />
              ))}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const READ_DIST = [
  { name: 'ztensor',     large: 2.14, mixed: 2.31, small: 1.91 },
  { name: 'safetensors', large: 1.28, mixed: 1.35, small: 1.35 },
  { name: 'pickle',      large: 1.43, mixed: 1.43, small: 1.57 },
  { name: 'npz',         large: 1.12, mixed: 1.12, small: 0.22 },
  { name: 'gguf',        large: 2.34, mixed: 2.29, small: 0.20 },
  { name: 'onnx',        large: 0.78, mixed: 0.78, small: 0.67 },
  { name: 'hdf5',        large: 1.33, mixed: 1.40, small: 0.16 },
];

const WRITE_DIST = [
  { name: 'ztensor',     large: 3.60, mixed: 3.65, small: 1.43 },
  { name: 'safetensors', large: 1.72, mixed: 1.75, small: 1.30 },
  { name: 'pickle',      large: 3.58, mixed: 3.65, small: 1.76 },
  { name: 'npz',         large: 2.39, mixed: 2.38, small: 0.50 },
  { name: 'gguf',        large: 3.81, mixed: 3.90, small: 1.01 },
  { name: 'onnx',        large: 0.29, mixed: 0.29, small: 0.34 },
  { name: 'hdf5',        large: 3.68, mixed: 3.67, small: 0.27 },
];

export function DistributionComparisonChart() {
  return <DistributionBarChart data={READ_DIST} yLabel="Read Throughput (GB/s)" yMax={3.0} />;
}


export function WriteDistributionChart() {
  return <DistributionBarChart data={WRITE_DIST} yLabel="Write Throughput (GB/s)" yMax={4.5} />;
}

// ---- Compression workload charts ----

const COMPRESSION_WORKLOADS = [
  { name: 'Dense fp32',       ratio: 0.92, readRaw: 2.31, readZstd: 0.45, writeRaw: 3.65, writeZstd: 0.72 },
  { name: 'Quantized int8',   ratio: 0.52, readRaw: 2.31, readZstd: 0.73, writeRaw: 3.65, writeZstd: 0.24 },
  { name: 'Pruned 80%',       ratio: 0.27, readRaw: 2.31, readZstd: 0.59, writeRaw: 3.65, writeZstd: 0.39 },
  { name: 'Ternary',          ratio: 0.25, readRaw: 2.31, readZstd: 0.90, writeRaw: 3.65, writeZstd: 0.45 },
];

function CompLegend({ items }) {
  const t = useTheme();
  return (
    <div style={{ display: 'flex', gap: 18, justifyContent: 'center', marginBottom: 4, fontSize: 13 }}>
      {items.map(([color, label]) => (
        <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 12, height: 12, background: color, borderRadius: 2, display: 'inline-block' }} />
          <span style={{ color: t.text, fontSize: 12 }}>{label}</span>
        </span>
      ))}
    </div>
  );
}

export function ZstdTradeoffChart({ height = 280 }) {
  const t = useTheme();
  const light = t.dark ? '#1E3A5F' : '#DBEAFE';
  const solid = t.dark ? '#3B82F6' : '#2563EB';
  const data = COMPRESSION_WORKLOADS.map(d => ({ ...d, original: 1.0 }));
  return (
    <div style={{ margin: '16px 0' }}>
      <CompLegend items={[[light, 'ztensor'], [solid, 'ztensor + zstd-3']]} />
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} barSize={BAR_WIDTH}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis dataKey="name" tick={{ fill: t.textMuted, fontSize: 12 }} axisLine={{ stroke: t.axis }} tickLine={false} />
          <YAxis
            tick={{ fill: t.textMuted, fontSize: 12 }}
            axisLine={false} tickLine={false}
            label={{ value: '% of original size', angle: -90, position: 'insideLeft', offset: 12, fill: t.textMuted, fontSize: 12 }}
            domain={[0, 1.05]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} width={50}
          />
          <Tooltip content={({ active, payload, label }) => {
            if (!active || !payload?.length) return null;
            const ratio = payload.find(p => p.dataKey === 'ratio')?.value;
            return (
              <div style={{ background: t.tooltipBg, border: `1px solid ${t.tooltipBorder}`, borderRadius: 8, padding: '10px 14px', fontSize: 13, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
                <p style={{ margin: 0, fontWeight: 600, color: t.text, fontSize: 12 }}>{label}</p>
                <p style={{ margin: '3px 0 0', color: solid, fontSize: 13 }}>
                  Compressed: <strong>{(ratio * 100).toFixed(0)}%</strong> of original
                </p>
                <p style={{ margin: '3px 0 0', color: t.textMuted, fontSize: 13 }}>
                  Saved: <strong>{((1 - ratio) * 100).toFixed(0)}%</strong>
                </p>
              </div>
            );
          }} />
          <Bar dataKey="original" name="ztensor" fill={light} radius={[4, 4, 0, 0]} />
          <Bar dataKey="ratio" name="ztensor + zstd-3" fill={solid} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CompThroughputSub({ data, rawKey, zstdKey, yMax, title }) {
  const t = useTheme();
  const light = t.dark ? '#1E3A5F' : '#DBEAFE';
  const solid = t.dark ? '#3B82F6' : '#2563EB';
  return (
    <div style={{ flex: 1, minWidth: 280 }}>
      <p style={{ textAlign: 'center', margin: '0 0 2px', fontSize: 12, fontWeight: 600, color: t.textMuted }}>{title}</p>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 4 }} barSize={BAR_WIDTH}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis dataKey="name" tick={{ fill: t.textMuted, fontSize: 11 }} axisLine={{ stroke: t.axis }} tickLine={false} />
          <YAxis
            tick={{ fill: t.textMuted, fontSize: 11 }}
            axisLine={false} tickLine={false}
            label={{ value: 'GB/s', angle: -90, position: 'insideLeft', offset: 8, fill: t.textMuted, fontSize: 11 }}
            domain={[0, yMax]} width={40}
          />
          <Tooltip content={<ChartTooltip suffix=" GB/s" />} />
          <Bar dataKey={rawKey} name="ztensor" fill={light} radius={[4, 4, 0, 0]} />
          <Bar dataKey={zstdKey} name="ztensor + zstd-3" fill={solid} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CompressionThroughputChart() {
  const t = useTheme();
  const light = t.dark ? '#1E3A5F' : '#DBEAFE';
  const solid = t.dark ? '#3B82F6' : '#2563EB';
  return (
    <div style={{ margin: '16px 0' }}>
      <CompLegend items={[[light, 'ztensor'], [solid, 'ztensor + zstd-3']]} />
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <CompThroughputSub data={COMPRESSION_WORKLOADS} rawKey="readRaw" zstdKey="readZstd" yMax={2.6} title="Read" />
        <CompThroughputSub data={COMPRESSION_WORKLOADS} rawKey="writeRaw" zstdKey="writeZstd" yMax={4.0} title="Write" />
      </div>
    </div>
  );
}

// ---- Data ----

const READ_MIXED = [
  { size: '128MB', ztensor: 2.65, safetensors: 1.82, pickle: 1.95, hdf5: 1.82, gguf: 2.65 },
  { size: '512MB', ztensor: 2.63, safetensors: 1.85, pickle: 1.89, hdf5: 1.80, gguf: 2.65 },
  { size: '1GB',   ztensor: 2.63, safetensors: 1.90, pickle: 1.90, hdf5: 1.90, gguf: 2.65 },
  { size: '2GB',   ztensor: 2.61, safetensors: 1.95, pickle: 1.99, hdf5: 1.96, gguf: 2.64 },
];

const READ_LARGE = [
  { size: '128MB', ztensor: 2.57, safetensors: 2.21, pickle: 2.15, hdf5: 1.92, gguf: 2.55 },
  { size: '512MB', ztensor: 2.57, safetensors: 2.26, pickle: 2.17, hdf5: 1.86, gguf: 2.61 },
  { size: '1GB',   ztensor: 2.64, safetensors: 2.29, pickle: 2.25, hdf5: 1.93, gguf: 2.59 },
  { size: '2GB',   ztensor: 2.66, safetensors: 2.28, pickle: 2.19, hdf5: 1.94, gguf: 2.63 },
];

const READ_SMALL = [
  { size: '128MB', ztensor: 1.78, safetensors: 1.14, pickle: 1.58, hdf5: 0.14, gguf: 0.21 },
  { size: '512MB', ztensor: 1.78, safetensors: 1.37, pickle: 1.51, hdf5: 0.16, gguf: 0.20 },
  { size: '1GB',   ztensor: 1.80, safetensors: 1.37, pickle: 1.46, hdf5: 0.16, gguf: 0.20 },
  { size: '2GB',   ztensor: 1.83, safetensors: 1.42, pickle: 1.56, hdf5: 0.17, gguf: 0.21 },
];

const WRITE_MIXED = [
  { size: '128MB', ztensor: 3.69, safetensors: 2.05, pickle: 3.79, hdf5: 3.83, gguf: 3.92 },
  { size: '512MB', ztensor: 3.66, safetensors: 1.78, pickle: 3.65, hdf5: 3.44, gguf: 3.92 },
  { size: '1GB',   ztensor: 3.63, safetensors: 1.77, pickle: 3.63, hdf5: 3.62, gguf: 3.84 },
  { size: '2GB',   ztensor: 3.68, safetensors: 1.77, pickle: 3.66, hdf5: 3.63, gguf: 3.89 },
];

const SELECTIVE_2GB = [
  { name: 'ztensor',     id: 'ztensor',     value: 0.148 },
  { name: 'safetensors', id: 'safetensors', value: 0.136 },
  { name: 'hdf5',        id: 'hdf5',        value: 0.148 },
  { name: 'gguf',        id: 'gguf',        value: 0.114 },
  { name: 'pickle',      id: 'pickle',      value: 0.820 },
];

const ZEROCOPY = [
  { name: '512MB', 'copy=False': 2.63, 'copy=True': 1.85 },
  { name: '1GB',   'copy=False': 2.63, 'copy=True': 1.90 },
  { name: '2GB',   'copy=False': 2.61, 'copy=True': 1.95 },
];

const COMPRESSION = [
  { name: 'Random float32',     size: 1893 },
  { name: 'Structured weights', size: 1519 },
];

const CROSS_FORMAT_READ = [
  { name: '.zt',          ztensor: 2.50, native: null },
  { name: '.safetensors', ztensor: 2.27, native: 1.48 },
  { name: '.pt',          ztensor: 2.26, native: 1.44 },
  { name: '.npz',         ztensor: 2.35, native: 1.15 },
  { name: '.gguf',        ztensor: 2.29, native: 2.34 },
  { name: '.onnx',        ztensor: 2.28, native: 0.79 },
  { name: '.h5',          ztensor: 2.41, native: 1.48 },
];

// ---- Exported charts ----

export function ReadMixedChart() {
  return <ThroughputChart data={READ_MIXED} formats={['ztensor', 'safetensors', 'pickle', 'hdf5', 'gguf']} />;
}

export function ReadLargeChart() {
  return <ThroughputChart data={READ_LARGE} formats={['ztensor', 'safetensors', 'pickle', 'hdf5', 'gguf']} />;
}

export function ReadSmallChart() {
  return <ThroughputChart data={READ_SMALL} formats={['ztensor', 'safetensors', 'pickle', 'hdf5', 'gguf']} />;
}

export function WriteMixedChart() {
  return <ThroughputChart data={WRITE_MIXED} formats={['ztensor', 'safetensors', 'pickle', 'hdf5', 'gguf']} yLabel="Write Throughput (GB/s)" />;
}

export function SelectiveLoadingChart() {
  return <SelectiveBarChart data={SELECTIVE_2GB} />;
}

export function ZeroCopyChart() {
  return <ZeroCopyBarChart data={ZEROCOPY} />;
}

export function CompressionChart() {
  return <CompressionBarChart data={COMPRESSION} />;
}

export function CrossFormatChart() {
  return <CrossFormatBarChart data={CROSS_FORMAT_READ} />;
}
