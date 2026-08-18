import type { ReactNode } from 'react'

export const ENTITY_COLORS: Record<string, { bg: string; border: string; text: string; dot: string }> = {
  Project: { bg: 'bg-sky-50', border: 'border-sky-300', text: 'text-sky-700', dot: '#38bdf8' },
  Component: { bg: 'bg-violet-50', border: 'border-violet-300', text: 'text-violet-700', dot: '#a78bfa' },
  Supplier: { bg: 'bg-amber-50', border: 'border-amber-300', text: 'text-amber-700', dot: '#fbbf24' },
  Contract: { bg: 'bg-emerald-50', border: 'border-emerald-300', text: 'text-emerald-700', dot: '#34d399' },
  Risk: { bg: 'bg-rose-50', border: 'border-rose-300', text: 'text-rose-700', dot: '#fb7185' },
  Incident: { bg: 'bg-orange-50', border: 'border-orange-300', text: 'text-orange-700', dot: '#fb923c' },
  Employee: { bg: 'bg-teal-50', border: 'border-teal-300', text: 'text-teal-700', dot: '#2dd4bf' },
  Unknown: { bg: 'bg-slate-500/25', border: 'border-slate-400/60', text: 'text-slate-800', dot: '#94a3b8' },
}

export function entityColor(type: string) {
  return ENTITY_COLORS[type] ?? ENTITY_COLORS.Unknown
}

export function Card({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={`rounded-xl border border-slate-200 bg-white backdrop-blur-sm shadow-xl shadow-slate-900/10 ${className}`}
    >
      {children}
    </div>
  )
}

export function SectionTitle({
  icon,
  children,
  right,
}: {
  icon?: ReactNode
  children: ReactNode
  right?: ReactNode
}) {
  return (
    <div className="flex items-center justify-between gap-3 mb-3">
      <div className="flex items-center gap-2 text-[13px] font-semibold uppercase tracking-[0.13em] text-slate-600">
        {icon}
        {children}
      </div>
      {right}
    </div>
  )
}

export function Badge({
  children,
  tone = 'slate',
  className = '',
}: {
  children: ReactNode
  tone?: 'slate' | 'sky' | 'emerald' | 'amber' | 'rose' | 'violet'
  className?: string
}) {
  const tones = {
    slate: 'bg-slate-500/20 text-slate-800 border-slate-400/50',
    sky: 'bg-sky-50 text-sky-700 border-sky-300',
    emerald: 'bg-emerald-50 text-emerald-700 border-emerald-300',
    amber: 'bg-amber-50 text-amber-700 border-amber-300',
    rose: 'bg-rose-50 text-rose-700 border-rose-300',
    violet: 'bg-violet-50 text-violet-700 border-violet-300',
  }
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[12px] font-medium ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

export function TypePill({ type }: { type: string }) {
  const color = entityColor(type)
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[12px] font-medium ${color.bg} ${color.border} ${color.text}`}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color.dot }} />
      {type}
    </span>
  )
}

export function Stat({
  label,
  value,
  sub,
  tone = 'neutral',
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: 'neutral' | 'good' | 'bad'
}) {
  const toneClass =
    tone === 'good' ? 'text-emerald-700' : tone === 'bad' ? 'text-rose-700' : 'text-slate-900'
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
      <div className="text-[12px] uppercase tracking-wider text-slate-600">{label}</div>
      <div className={`mt-0.5 text-2xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
      {sub ? <div className="text-[12px] text-slate-600">{sub}</div> : null}
    </div>
  )
}

export function Spinner({ className = 'h-4 w-4' }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.2" strokeWidth="3" />
      <path
        d="M22 12a10 10 0 0 1-10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function ScoreBar({ value, tone }: { value: number; tone: 'basic' | 'semantic' }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const color = tone === 'basic' ? 'bg-amber-400/70' : 'bg-emerald-400/70'
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
      <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 px-6 py-12 text-center">
      <div className="text-[15px] font-medium text-slate-800">{title}</div>
      <div className="mt-1 max-w-md text-[13px] text-slate-600">{body}</div>
    </div>
  )
}

export function ms(value?: number) {
  if (value === undefined || value === null) return '—'
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`
  return `${Math.round(value)}ms`
}

export function pct(value?: number) {
  if (value === undefined || value === null) return '—'
  return `${(value * 100).toFixed(0)}%`
}
