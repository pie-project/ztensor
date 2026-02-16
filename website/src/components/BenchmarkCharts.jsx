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

export function CrossFormatBarChart({ data, height = 340 }) {
  const t = useTheme();
  const zcBlue = t.dark ? '#2563EB' : '#1D4ED8';
  const copyBlue = t.dark ? '#93C5FD' : '#93C5FD';
  const zcGray = t.dark ? '#4B5563' : '#9CA3AF';
  const copyGray = t.dark ? '#9CA3AF' : '#D1D5DB';
  return (
    <div style={{ margin: '16px 0' }}>
      <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginBottom: 4, fontSize: 12, flexWrap: 'wrap' }}>
        {[
          [zcBlue, 'ztensor'],
          [copyBlue, 'ztensor (zc off)'],
          [zcGray, 'ref. zero-copy'],
          [copyGray, 'ref. copy'],
        ].map(([color, label]) => (
          <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 12, height: 12, background: color, borderRadius: 3, display: 'inline-block' }} />
            <span style={{ color: t.text, fontWeight: label.startsWith('ztensor') ? 600 : 400 }}>{label}</span>
          </span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} barSize={16}>
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
          <Bar dataKey="ztensorZc" name="ztensor" fill={zcBlue} radius={[4, 4, 0, 0]} />
          <Bar dataKey="ztensor" name="ztensor (zc off)" fill={copyBlue} radius={[4, 4, 0, 0]} />
          <Bar dataKey="nativeZc" name="Ref. zero-copy" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.nativeZc != null ? zcGray : 'transparent'} />
            ))}
          </Bar>
          <Bar dataKey="native" name="Ref. copy" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.native != null ? copyGray : 'transparent'} />
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
                <Cell key={i} fill={entry.name.startsWith('ztensor') ? blue[key] : gray[key]} />
              ))}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const READ_DIST = [
  { name: 'ztensor',              large: 2.08, mixed: 2.02, small: 1.76, zc: true },
  { name: 'ztensor\n(zc off)',    large: 1.25, mixed: 1.31, small: 1.46 },
  { name: 'safetensors',          large: 1.23, mixed: 1.32, small: 1.35 },
  { name: 'pickle',               large: 1.25, mixed: 1.36, small: 1.40 },
  { name: 'npz',                  large: 1.05, mixed: 1.06, small: 0.22 },
  { name: 'gguf',                 large: 2.32, mixed: 2.31, small: 0.21, zc: true },
  { name: 'gguf\n(zc off)',       large: 1.40, mixed: 1.40, small: 0.20 },
  { name: 'onnx',                 large: 0.73, mixed: 0.75, small: 0.65 },
  { name: 'hdf5',                 large: 1.28, mixed: 1.33, small: 0.16 },
];

const WRITE_DIST = [
  { name: 'ztensor',     large: 3.62, mixed: 3.65, small: 1.42 },
  { name: 'safetensors', large: 1.72, mixed: 1.77, small: 1.48 },
  { name: 'pickle',      large: 3.62, mixed: 3.68, small: 2.00 },
  { name: 'npz',         large: 2.40, mixed: 2.40, small: 0.51 },
  { name: 'gguf',        large: 3.85, mixed: 3.86, small: 1.06 },
  { name: 'onnx',        large: 0.28, mixed: 0.29, small: 0.32 },
  { name: 'hdf5',        large: 3.67, mixed: 3.69, small: 0.27 },
];

export function DistributionComparisonChart() {
  return <DistributionBarChart data={READ_DIST} yLabel="Read Throughput (GB/s)" yMax={2.8} />;
}


export function WriteDistributionChart() {
  return <DistributionBarChart data={WRITE_DIST} yLabel="Write Throughput (GB/s)" yMax={4.5} />;
}

// ---- Compression workload charts ----

const COMPRESSION_WORKLOADS = [
  { name: 'Dense fp32',       ratio: 0.92, readRaw: 1.31, readZstd: 0.45, writeRaw: 3.65, writeZstd: 0.72 },
  { name: 'Quantized int8',   ratio: 0.52, readRaw: 1.31, readZstd: 0.73, writeRaw: 3.65, writeZstd: 0.24 },
  { name: 'Pruned 80%',       ratio: 0.27, readRaw: 1.31, readZstd: 0.59, writeRaw: 3.65, writeZstd: 0.39 },
  { name: 'Ternary',          ratio: 0.25, readRaw: 1.31, readZstd: 0.90, writeRaw: 3.65, writeZstd: 0.45 },
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
        <CompThroughputSub data={COMPRESSION_WORKLOADS} rawKey="readRaw" zstdKey="readZstd" yMax={1.5} title="Read" />
        <CompThroughputSub data={COMPRESSION_WORKLOADS} rawKey="writeRaw" zstdKey="writeZstd" yMax={4.0} title="Write" />
      </div>
    </div>
  );
}

// ---- Data ----

const READ_MIXED = [
  { size: '128MB', ztensor: 1.29, safetensors: 1.28, pickle: 1.32, hdf5: 1.31, gguf: 1.35 },
  { size: '512MB', ztensor: 1.31, safetensors: 1.32, pickle: 1.36, hdf5: 1.33, gguf: 1.40 },
  { size: '1GB',   ztensor: 1.30, safetensors: 1.29, pickle: 1.27, hdf5: 1.34, gguf: 1.38 },
  { size: '2GB',   ztensor: 1.23, safetensors: 1.23, pickle: 1.28, hdf5: 1.34, gguf: 1.38 },
];

const READ_LARGE = [
  { size: '128MB', ztensor: 1.38, safetensors: 1.35, pickle: 1.28, hdf5: 1.36, gguf: 1.41 },
  { size: '512MB', ztensor: 1.25, safetensors: 1.23, pickle: 1.25, hdf5: 1.28, gguf: 1.40 },
  { size: '1GB',   ztensor: 1.31, safetensors: 1.29, pickle: 1.37, hdf5: 1.34, gguf: 1.37 },
  { size: '2GB',   ztensor: 1.33, safetensors: 1.32, pickle: 1.37, hdf5: 1.36, gguf: 1.37 },
];

const READ_SMALL = [
  { size: '128MB', ztensor: 1.46, safetensors: 1.41, pickle: 1.68, hdf5: 0.15, gguf: 0.21 },
  { size: '512MB', ztensor: 1.46, safetensors: 1.35, pickle: 1.40, hdf5: 0.16, gguf: 0.20 },
  { size: '1GB',   ztensor: 1.46, safetensors: 1.31, pickle: 1.44, hdf5: 0.16, gguf: 0.20 },
  { size: '2GB',   ztensor: 1.44, safetensors: 1.31, pickle: 1.42, hdf5: 0.16, gguf: 0.20 },
];

const WRITE_MIXED = [
  { size: '128MB', ztensor: 3.77, safetensors: 2.01, pickle: 3.82, hdf5: 3.86, gguf: 3.92 },
  { size: '512MB', ztensor: 3.65, safetensors: 1.77, pickle: 3.68, hdf5: 3.69, gguf: 3.86 },
  { size: '1GB',   ztensor: 3.65, safetensors: 1.69, pickle: 3.65, hdf5: 3.61, gguf: 3.86 },
  { size: '2GB',   ztensor: 3.57, safetensors: 1.68, pickle: 3.58, hdf5: 3.54, gguf: 3.78 },
];

const SELECTIVE_2GB = [
  { name: 'ztensor',     id: 'ztensor',     value: 0.176 },
  { name: 'safetensors', id: 'safetensors', value: 0.177 },
  { name: 'hdf5',        id: 'hdf5',        value: 0.171 },
  { name: 'gguf',        id: 'gguf',        value: 0.161 },
  { name: 'pickle',      id: 'pickle',      value: 1.013 },
];

const ZEROCOPY = [
  { name: '512MB', 'copy=False': 2.08, 'copy=True': 1.31 },
  { name: '1GB',   'copy=False': 2.12, 'copy=True': 1.30 },
  { name: '2GB',   'copy=False': 2.14, 'copy=True': 1.30 },
];

const COMPRESSION = [
  { name: 'Random float32',     size: 1893 },
  { name: 'Structured weights', size: 1519 },
];

const CROSS_FORMAT_READ = [
  // ztensorZc: ztensor copy=False (zero-copy mmap), ztensor: copy=True
  // nativeZc: native lib zero-copy where available (gguf mmap, safetensors safe_open)
  { name: '.zt',          ztensorZc: 2.19, ztensor: 1.37, native: null, nativeZc: null },
  { name: '.safetensors', ztensorZc: 2.19, ztensor: 1.46, native: 1.33, nativeZc: 1.35 },
  { name: '.pt',          ztensorZc: 2.04, ztensor: 1.33, native: 0.89, nativeZc: null },
  { name: '.npz',         ztensorZc: 2.11, ztensor: 1.41, native: 1.04, nativeZc: null },
  { name: '.gguf',        ztensorZc: 2.11, ztensor: 1.38, native: 1.39, nativeZc: 2.15 },
  { name: '.onnx',        ztensorZc: 2.07, ztensor: 1.29, native: 0.76, nativeZc: null },
  { name: '.h5',          ztensorZc: 1.96, ztensor: 1.30, native: 1.35, nativeZc: null },
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
