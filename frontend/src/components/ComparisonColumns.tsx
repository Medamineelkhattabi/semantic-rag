import { useState } from 'react'
import {
  Boxes,
  ChevronDown,
  Clock,
  FileText,
  GitBranch,
  Layers,
  Route,
  Target,
} from 'lucide-react'
import type { BasicResult, SemanticResult } from '../types'
import { Badge, Card, SectionTitle, Stat, TypePill, ms } from './ui'

function AnswerBlock({ text, tone }: { text: string; tone: 'basic' | 'semantic' }) {
  const ring = tone === 'basic' ? 'border-amber-300' : 'border-emerald-300'
  return (
    <div className={`rounded-lg border ${ring} bg-slate-50 px-4 py-3`}>
      <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-slate-800">
        {text || '—'}
      </p>
    </div>
  )
}

function Collapsible({
  title,
  count,
  children,
  defaultOpen = false,
}: {
  title: string
  count?: number
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
      >
        <span className="text-[13px] font-semibold uppercase tracking-wider text-slate-600">
          {title}
          {count !== undefined ? <span className="ml-1.5 text-slate-600">({count})</span> : null}
        </span>
        <ChevronDown
          className={`h-3.5 w-3.5 text-slate-600 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {open ? <div className="border-t border-slate-200 px-3 py-3">{children}</div> : null}
    </div>
  )
}

function LatencyRow({ latency }: { latency: Record<string, number | undefined> }) {
  const entries = Object.entries(latency).filter(([key]) => key !== 'total_ms')
  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.map(([key, value]) => (
        <span
          key={key}
          className="rounded border border-slate-200 bg-slate-100 px-1.5 py-0.5 text-[12px] text-slate-600"
        >
          {key.replace(/_ms$/, '').replace(/_/g, ' ')}{' '}
          <span className="font-medium tabular-nums text-slate-800">{ms(value)}</span>
        </span>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ */
export function BasicColumn({ result }: { result: BasicResult }) {
  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-50 text-amber-700">
            <Layers className="h-4 w-4" />
          </span>
          <div>
            <h3 className="text-[15px] font-semibold text-slate-900">Basic RAG</h3>
            <p className="text-[12px] text-slate-600">
              chunk → embed → vector search → top-K → LLM
            </p>
          </div>
        </div>
        <Badge tone="amber">
          <Clock className="h-3 w-3" />
          {ms(result.latency.total_ms)}
        </Badge>
      </div>

      <AnswerBlock text={result.answer} tone="basic" />

      <div className="grid grid-cols-3 gap-2">
        <Stat label="Chunks" value={result.chunks.length} />
        <Stat label="Sources" value={result.sources.length} />
        <Stat label="Context" value={`${(result.context.length / 1000).toFixed(1)}k`} sub="chars" />
      </div>

      <LatencyRow latency={result.latency as unknown as Record<string, number>} />

      <div>
        <SectionTitle icon={<Boxes className="h-3.5 w-3.5" />}>
          Retrieved chunks &amp; similarity
        </SectionTitle>
        <div className="flex flex-col gap-2">
          {result.chunks.map((chunk) => (
            <div
              key={chunk.chunkId}
              className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 truncate">
                  <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded bg-slate-200 text-[11px] font-semibold text-slate-600">
                    {chunk.rank}
                  </span>
                  <span className="truncate font-mono text-[12px] text-slate-600">
                    {chunk.docId}
                  </span>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  <div className="h-1 w-14 overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-amber-400/70"
                      style={{ width: `${Math.max(0, Math.min(1, chunk.score)) * 100}%` }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-[12px] tabular-nums text-amber-700">
                    {chunk.score.toFixed(3)}
                  </span>
                </div>
              </div>
              <p className="mt-1.5 line-clamp-3 text-[13px] leading-relaxed text-slate-600">
                {chunk.text}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div>
        <SectionTitle icon={<FileText className="h-3.5 w-3.5" />}>Sources</SectionTitle>
        <div className="flex flex-wrap gap-1.5">
          {result.sources.map((source) => (
            <span
              key={source}
              className="rounded border border-amber-300 bg-amber-50 px-1.5 py-0.5 font-mono text-[12px] text-amber-700"
            >
              {source}
            </span>
          ))}
        </div>
      </div>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
export function SemanticColumn({ result }: { result: SemanticResult }) {
  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
            <GitBranch className="h-4 w-4" />
          </span>
          <div>
            <h3 className="text-[15px] font-semibold text-slate-900">Semantic RAG</h3>
            <p className="text-[12px] text-slate-600">
              extract → graph → link → traverse → expand → LLM
            </p>
          </div>
        </div>
        <Badge tone="emerald">
          <Clock className="h-3 w-3" />
          {ms(result.latency.total_ms)}
        </Badge>
      </div>

      <AnswerBlock text={result.answer} tone="semantic" />

      <div className="grid grid-cols-3 gap-2">
        <Stat label="Facts" value={result.relationships.length} sub={`${result.relationships.filter((r) => r.onPath).length} on path`} />
        <Stat label="Paths" value={result.paths.length} sub={`max ${Math.max(0, ...result.paths.map((p) => p.hops))} hops`} />
        <Stat label="Sources" value={result.sources.length} />
      </div>

      <LatencyRow latency={result.latency as unknown as Record<string, number>} />

      <div>
        <SectionTitle icon={<Target className="h-3.5 w-3.5" />}>
          Linked entities
          {result.targetTypes.length ? (
            <span className="ml-1 normal-case tracking-normal text-slate-600">
              · looking for {result.targetTypes.join(', ')}
            </span>
          ) : null}
        </SectionTitle>
        <div className="flex flex-wrap gap-1.5">
          {result.entities.map((entity) => (
            <span
              key={entity.id}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-100 px-2 py-1"
              title={`matched by ${entity.method} (${entity.score.toFixed(3)})`}
            >
              <TypePill type={entity.type} />
              <span className="text-[13px] font-medium text-slate-800">{entity.id}</span>
              <span className="font-mono text-[11px] text-slate-600">{entity.method}</span>
            </span>
          ))}
          {result.entities.length === 0 ? (
            <span className="text-[13px] text-slate-600">No entity matched this question.</span>
          ) : null}
        </div>
      </div>

      <div>
        <SectionTitle icon={<Route className="h-3.5 w-3.5" />}>Retrieval path</SectionTitle>
        <div className="flex flex-col gap-1.5">
          {result.paths.slice(0, 4).map((path, index) => (
            <div
              key={index}
              className={`rounded-lg border px-3 py-2 ${
                index === 0
                  ? 'border-emerald-300 bg-emerald-500/[0.06]'
                  : 'border-slate-200 bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-[13px] text-slate-800">{path.chain}</span>
                <span className="shrink-0 rounded bg-slate-200 px-1.5 py-0.5 text-[11px] text-slate-600">
                  {path.hops} hops
                </span>
              </div>
            </div>
          ))}
          {result.paths.length === 0 ? (
            <span className="text-[13px] text-slate-600">
              No connecting path — answered from the local neighbourhood.
            </span>
          ) : null}
        </div>
      </div>

      <Collapsible title="Relationships used" count={result.relationships.length}>
        <div className="flex max-h-64 flex-col gap-1 overflow-y-auto">
          {result.relationships.map((rel, index) => (
            <div
              key={index}
              className={`flex flex-wrap items-center gap-1.5 rounded px-2 py-1 text-[12px] ${
                rel.onPath ? 'bg-emerald-50' : ''
              }`}
            >
              {rel.onPath ? <span className="text-emerald-600">★</span> : null}
              <span className="font-mono text-slate-800">{rel.source}</span>
              <span className="rounded bg-slate-200 px-1 font-mono text-slate-600">
                {rel.relation}
              </span>
              <span className="font-mono text-slate-800">{rel.target}</span>
              {rel.docIds.map((docId) => (
                <span key={docId} className="font-mono text-[11px] text-slate-600">
                  [{docId}]
                </span>
              ))}
            </div>
          ))}
        </div>
      </Collapsible>

      <Collapsible title="Source provenance" count={result.passages.length}>
        <div className="flex max-h-64 flex-col gap-2 overflow-y-auto">
          {result.passages.map((passage, index) => (
            <div key={index} className="rounded border border-slate-200 bg-slate-50 px-2 py-1.5">
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-[12px] text-emerald-700">{passage.docId}</span>
                <span className="text-[12px] text-slate-600">· {passage.heading}</span>
              </div>
              <p className="mt-1 line-clamp-2 text-[12px] text-slate-600">{passage.text}</p>
            </div>
          ))}
        </div>
      </Collapsible>

      <div>
        <SectionTitle icon={<FileText className="h-3.5 w-3.5" />}>Sources</SectionTitle>
        <div className="flex flex-wrap gap-1.5">
          {result.sources.map((source) => (
            <span
              key={source}
              className="rounded border border-emerald-300 bg-emerald-50 px-1.5 py-0.5 font-mono text-[12px] text-emerald-700"
            >
              {source}
            </span>
          ))}
        </div>
      </div>
    </Card>
  )
}
