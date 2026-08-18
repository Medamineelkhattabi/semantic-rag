import { ArrowRight, Sparkles } from 'lucide-react'
import type { GraphPath } from '../types'
import { entityColor } from './ui'

/**
 * The headline visual: the exact entity chain the graph walked to answer.
 * e.g. Project Phoenix -> C-17 -> Alpha Precision Systems -> C-2048 -> R-17
 */
export function WhySemanticRag({
  path,
  basicSources,
  semanticSources,
}: {
  path: GraphPath | null
  basicSources: string[]
  semanticSources: string[]
}) {
  if (!path || path.nodes.length < 2) return null

  return (
    <div className="animate-rise overflow-hidden rounded-xl border border-emerald-400/25 bg-gradient-to-br from-emerald-500/[0.09] via-ink-850/80 to-sky-500/[0.06]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/6 px-5 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-emerald-300" />
          <h2 className="text-sm font-semibold text-slate-100">Why Semantic RAG?</h2>
          <span className="text-xs text-slate-400">
            The answer follows a {path.hops}-hop chain across{' '}
            {new Set(path.steps.flatMap((s) => s.docIds)).size} documents
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px]">
          <span className="rounded-md border border-amber-400/25 bg-amber-500/10 px-2 py-1 text-amber-300">
            Basic reached {basicSources.length} doc{basicSources.length === 1 ? '' : 's'}
          </span>
          <span className="rounded-md border border-emerald-400/25 bg-emerald-500/10 px-2 py-1 text-emerald-300">
            Semantic reached {semanticSources.length} doc{semanticSources.length === 1 ? '' : 's'}
          </span>
        </div>
      </div>

      {/* The chain ------------------------------------------------- */}
      <div className="overflow-x-auto px-5 py-6">
        <div className="flex min-w-max items-stretch gap-1">
          {path.nodes.map((node, index) => {
            const color = entityColor(node.type)
            const step = path.steps[index]
            return (
              <div key={`${node.id}-${index}`} className="flex items-stretch gap-1">
                <div
                  className={`flex min-w-[132px] flex-col justify-center rounded-lg border px-3 py-2.5 ${color.bg} ${color.border} ${
                    index === 0 ? 'pulse-ring' : ''
                  }`}
                  style={{ animationDelay: `${index * 90}ms` }}
                >
                  <div className="flex items-center gap-1.5">
                    <span
                      className="h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{ background: color.dot }}
                    />
                    <span className="text-[9px] uppercase tracking-wider text-slate-500">
                      {node.type}
                    </span>
                  </div>
                  <div className={`mt-0.5 text-sm font-semibold ${color.text}`}>{node.id}</div>
                  {node.label && node.label !== node.id ? (
                    <div className="truncate text-[10px] text-slate-500" title={node.label}>
                      {node.label}
                    </div>
                  ) : null}
                </div>

                {step ? (
                  <div className="flex flex-col items-center justify-center px-1">
                    <div className="whitespace-nowrap rounded bg-white/5 px-1.5 py-0.5 font-mono text-[9px] text-slate-400">
                      {step.relation}
                    </div>
                    <ArrowRight className="mt-1 h-3.5 w-3.5 text-slate-600" />
                  </div>
                ) : null}
              </div>
            )
          })}
        </div>

        <div className="mt-5 flex flex-col gap-2 text-xs">
          <div className="flex gap-2">
            <span className="shrink-0 font-medium text-amber-300/80">Basic RAG asks:</span>
            <span className="text-slate-400">
              &ldquo;Which text is most similar to my question?&rdquo;
            </span>
          </div>
          <div className="flex gap-2">
            <span className="shrink-0 font-medium text-emerald-300/80">Semantic RAG asks:</span>
            <span className="text-slate-400">
              &ldquo;Which entities, facts and relationships are relevant to my question?&rdquo;
            </span>
          </div>
        </div>
      </div>

      {/* Per-hop provenance ---------------------------------------- */}
      <div className="border-t border-white/6 bg-black/15 px-5 py-3">
        <div className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
          Provenance for each hop
        </div>
        <div className="flex flex-col gap-1.5">
          {path.steps.map((step, index) => (
            <div key={index} className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="font-mono text-slate-300">{step.phrase}</span>
              {step.docIds.map((docId) => (
                <span
                  key={docId}
                  className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[9px] text-slate-400"
                >
                  {docId}
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
