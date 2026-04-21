'use client';

import { useState } from 'react';
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend,
} from 'recharts';
import { SleepMetrics, SleepPatient, SleepReportData } from '@/lib/api';

// ─── Metric display config ───────────────────────────────────────────────────

const METRIC_LABELS: Record<string, string> = {
  ahi: 'AHI',
  rdi: 'RDI',
  sleep_efficiency: 'Sleep Efficiency',
  total_sleep_time: 'Total Sleep Time',
  spo2_min: 'SpO₂ Min',
  spo2_avg: 'SpO₂ Avg',
  waso: 'WASO',
  sleep_onset_latency: 'Sleep Onset',
  rem_pct: 'REM',
  n1_pct: 'N1',
  n2_pct: 'N2',
  n3_pct: 'N3 (Deep)',
  rem_latency: 'REM Latency',
  arousal_index: 'Arousal Index',
};

// Keys shown as top summary cards
const SUMMARY_KEYS = ['ahi', 'sleep_efficiency', 'spo2_min', 'total_sleep_time'];

// Keys shown in the sleep stage bar chart
const STAGE_KEYS = ['n1_pct', 'n2_pct', 'n3_pct', 'rem_pct'];
const STAGE_COLORS = ['#94a3b8', '#64748b', '#3b82f6', '#8b5cf6'];

// Keys shown in trend lines (multi-report)
const TREND_KEYS = ['ahi', 'sleep_efficiency', 'spo2_min'];

function formatValue(key: string, value: number, unit: string): string {
  if (key === 'total_sleep_time') {
    const h = Math.floor(value / 60);
    const m = Math.round(value % 60);
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
  }
  if (unit === '%') return `${value.toFixed(1)}%`;
  if (unit === '/hr') return `${value.toFixed(1)}/hr`;
  if (unit === 'min') return `${value.toFixed(0)} min`;
  return `${value.toFixed(1)}${unit ? ' ' + unit : ''}`;
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function MetricCard({ metricKey, value, unit, flagged, flagNote }: {
  metricKey: string;
  value: number;
  unit: string;
  flagged: boolean;
  flagNote?: string;
}) {
  const label = METRIC_LABELS[metricKey] ?? metricKey.replace(/_/g, ' ');
  const displayVal = formatValue(metricKey, value, unit);

  return (
    <div className={`rounded-xl p-3 flex flex-col gap-1 border ${
      flagged
        ? 'bg-amber-50 border-amber-200 dark:bg-amber-950/30 dark:border-amber-700'
        : 'bg-white/60 border-white/40 dark:bg-white/5 dark:border-white/10'
    }`}>
      <span className="text-xs text-slate-500 dark:text-slate-400 font-medium uppercase tracking-wide">
        {label}
      </span>
      <span className={`text-xl font-bold ${
        flagged ? 'text-amber-600 dark:text-amber-400' : 'text-slate-800 dark:text-slate-100'
      }`}>
        {displayVal}
      </span>
      {flagged && flagNote && (
        <span className="text-xs text-amber-600 dark:text-amber-400 leading-tight">
          {flagNote}
        </span>
      )}
    </div>
  );
}

function StageChart({ report }: { report: SleepReportData }) {
  const data = STAGE_KEYS
    .filter((k) => k in report.metrics)
    .map((k, i) => ({
      name: METRIC_LABELS[k] ?? k,
      value: report.metrics[k].value,
      color: STAGE_COLORS[i],
    }));

  if (data.length < 2) return null;

  return (
    <div>
      <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
        Sleep Stages
      </p>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" horizontal={false} />
          <XAxis type="number" unit="%" tick={{ fontSize: 11 }} stroke="rgba(148,163,184,0.5)" />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={70} stroke="rgba(148,163,184,0.5)" />
          <Tooltip
            formatter={(v) => [typeof v === 'number' ? `${v.toFixed(1)}%` : String(v), '']}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function TrendChart({ patient }: { patient: SleepPatient }) {
  if (patient.reports.length < 2) return null;

  const availableKeys = TREND_KEYS.filter((k) =>
    patient.reports.some((r) => k in r.metrics)
  );
  if (availableKeys.length === 0) return null;

  const chartData = patient.reports.map((r, i) => {
    const point: Record<string, number | string> = {
      label: r.date ? r.date.slice(0, 10) : `Report ${i + 1}`,
    };
    availableKeys.forEach((k) => {
      if (k in r.metrics) point[k] = r.metrics[k].value;
    });
    return point;
  });

  const LINE_COLORS = ['#ef4444', '#10b981', '#3b82f6'];

  return (
    <div>
      <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
        Trends Over Time
      </p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={chartData} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="label" tick={{ fontSize: 11 }} stroke="rgba(148,163,184,0.5)" />
          <YAxis tick={{ fontSize: 11 }} stroke="rgba(148,163,184,0.5)" />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {availableKeys.map((k, i) => (
            <Line
              key={k}
              type="monotone"
              dataKey={k}
              name={METRIC_LABELS[k] ?? k}
              stroke={LINE_COLORS[i % LINE_COLORS.length]}
              strokeWidth={2}
              dot={{ r: 4 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function PatientPanel({ patient }: { patient: SleepPatient }) {
  const [reportIndex, setReportIndex] = useState(0);
  const report = patient.reports[reportIndex];
  if (!report) return null;

  const summaryMetrics = SUMMARY_KEYS
    .filter((k) => k in report.metrics)
    .map((k) => ({ key: k, ...report.metrics[k] }));

  return (
    <div className="space-y-4">
      {/* Patient header + report selector */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <p className="font-semibold text-slate-800 dark:text-slate-100">{patient.name}</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">{report.report_type}</p>
        </div>
        {patient.reports.length > 1 && (
          <div className="flex gap-1">
            {patient.reports.map((r, i) => (
              <button
                key={i}
                onClick={() => setReportIndex(i)}
                className={`text-xs px-2 py-1 rounded-lg border transition-colors ${
                  i === reportIndex
                    ? 'bg-blue-500 text-white border-blue-500'
                    : 'border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'
                }`}
              >
                {r.date ? r.date.slice(0, 10) : `Report ${i + 1}`}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Summary metric cards */}
      {summaryMetrics.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {summaryMetrics.map(({ key, value, unit, flagged, flag_note }) => (
            <MetricCard
              key={key}
              metricKey={key}
              value={value!}
              unit={unit}
              flagged={flagged}
              flagNote={flag_note}
            />
          ))}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <StageChart report={report} />
        <TrendChart patient={patient} />
      </div>
    </div>
  );
}

// ─── Main export ─────────────────────────────────────────────────────────────

export function SleepDashboard({ metrics }: { metrics: SleepMetrics }) {
  const [open, setOpen] = useState(true);

  return (
    <div className="mt-3 rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-slate-800/50 backdrop-blur-sm overflow-hidden">
      {/* Header / toggle */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/20 dark:hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">📊</span>
          <span className="font-semibold text-slate-800 dark:text-slate-100 text-sm">
            Sleep Report Dashboard
          </span>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {metrics.total_reports} report{metrics.total_reports !== 1 ? 's' : ''}
            {metrics.total_patients > 1 && `, ${metrics.total_patients} patients`}
          </span>
        </div>
        <span className="text-slate-400 text-xs">{open ? '▲' : '▼'}</span>
      </button>

      {/* Body */}
      {open && (
        <div className="px-4 pb-4 space-y-6 border-t border-white/20 dark:border-white/10 pt-4">
          {metrics.patients.map((patient, i) => (
            <PatientPanel key={i} patient={patient} />
          ))}
        </div>
      )}
    </div>
  );
}
