import type { ReactNode } from 'react'

export const ENTITY_COLORS: Record<string, { bg: string; border: string; text: string; dot: string }> = {
  Project: { bg: 'bg-sky-500/12', border: 'border-sky-400/40', text: 'text-sky-300', dot: '#38bdf8' },
  Component: { bg: 'bg-violet-500/12', border: 'border-violet-400/40', text: 'text-violet-300', dot: '#a78bfa' },
  Supplier: { bg: 'bg-amber-500/12', border: 'border-amber-400/40', text: 'text-amber-300', dot: '#fbbf24' },
  Contract: { bg: 'bg-emerald-500/12', border: 'border-emerald-400/40', text: 'text-emerald-300', dot: '#34d399' },
  Risk: { bg: 'bg-rose-500/12', border: 'border-rose-400/40', text: 'text-rose-300', dot: '#fb7185' },
  Incident: { bg: 'bg-orange-500/12', border: 'border-orange-400/40', text: 'text-orange-300', dot: '#fb923c' },
  Employee: { bg: 'bg-teal-500/12', border: 'border-teal-400/40', text: 'text-teal-300', dot: '#2dd4bf' },
  Unknown: { bg: 'bg-slate-500/12', border: 'border-slate-400/40', text: 'text-slate-300', dot: '#94a3b8' },
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
      className={`rounded-xl border border-white/8 bg-ink-850/70 backdrop-blur-sm shadow-lg shadow-black/25 ${className}`}
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
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.13em] text-slate-400">
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
    slate: 'bg-slate-500/12 text-slate-300 border-slate-400/25',
    sky: 'bg-sky-500/12 text-sky-300 border-sky-400/30',
    emerald: 'bg-emerald-500/12 text-emerald-300 border-emerald-400/30',
    amber: 'bg-amber-500/12 text-amber-300 border-amber-400/30',
    rose: 'bg-rose-500/12 text-rose-300 border-rose-400/30',
    violet: 'bg-violet-500/12 text-violet-300 border-violet-400/30',
  }
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

export function TypePill({ type }: { type: string }) {
  const color = entityColor(type)
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${color.bg} ${color.border} ${color.text}`}
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
    tone === 'good' ? 'text-emerald-300' : tone === 'bad' ? 'text-rose-300' : 'text-slate-100'
  return (
    <div className="rounded-lg border border-white/6 bg-white/[0.02] px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`mt-0.5 text-lg font-semibold tabular-nums ${toneClass}`}>{value}</div>
      {sub ? <div className="text-[10px] text-slate-500">{sub}</div> : null}
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
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/6">
      <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-white/10 px-6 py-12 text-center">
      <div className="text-sm font-medium text-slate-300">{title}</div>
      <div className="mt-1 max-w-md text-xs text-slate-500">{body}</div>
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
