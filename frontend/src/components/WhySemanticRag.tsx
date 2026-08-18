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
    <div className="animate-rise overflow-hidden rounded-xl border border-emerald-300 bg-gradient-to-br from-emerald-50 via-white to-sky-50">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-emerald-700" />
          <h2 className="text-[15px] font-semibold text-slate-900">Why Semantic RAG?</h2>
          <span className="text-[13px] text-slate-600">
            The answer follows a {path.hops}-hop chain across{' '}
            {new Set(path.steps.flatMap((s) => s.docIds)).size} documents
          </span>
        </div>
        <div className="flex items-center gap-2 text-[13px]">
          <span className="rounded-md border border-amber-300 bg-amber-50 px-2 py-1 text-amber-700">
            Basic reached {basicSources.length} doc{basicSources.length === 1 ? '' : 's'}
          </span>
          <span className="rounded-md border border-emerald-300 bg-emerald-50 px-2 py-1 text-emerald-700">
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
                    <span className="text-[11px] uppercase tracking-wider text-slate-600">
                      {node.type}
                    </span>
                  </div>
                  <div className={`mt-0.5 text-[15px] font-semibold ${color.text}`}>{node.id}</div>
                  {node.label && node.label !== node.id ? (
                    <div className="truncate text-[12px] text-slate-600" title={node.label}>
                      {node.label}
                    </div>
                  ) : null}
                </div>

                {step ? (
                  <div className="flex flex-col items-center justify-center px-1">
                    <div className="whitespace-nowrap rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[11px] text-slate-600">
                      {step.relation}
                    </div>
                    <ArrowRight className="mt-1 h-3.5 w-3.5 text-slate-600" />
                  </div>
                ) : null}
              </div>
            )
          })}
        </div>

        <div className="mt-5 flex flex-col gap-2 text-[13px]">
          <div className="flex gap-2">
            <span className="shrink-0 font-medium text-amber-700">Basic RAG asks:</span>
            <span className="text-slate-600">
              &ldquo;Which text is most similar to my question?&rdquo;
            </span>
          </div>
          <div className="flex gap-2">
            <span className="shrink-0 font-medium text-emerald-700">Semantic RAG asks:</span>
            <span className="text-slate-600">
              &ldquo;Which entities, facts and relationships are relevant to my question?&rdquo;
            </span>
          </div>
        </div>
      </div>

      {/* Per-hop provenance ---------------------------------------- */}
      <div className="border-t border-slate-200 bg-slate-50 px-5 py-3">
        <div className="mb-2 text-[12px] uppercase tracking-wider text-slate-600">
          Provenance for each hop
        </div>
        <div className="flex flex-col gap-1.5">
          {path.steps.map((step, index) => (
            <div key={index} className="flex flex-wrap items-center gap-2 text-[13px]">
              <span className="font-mono text-slate-800">{step.phrase}</span>
              {step.docIds.map((docId) => (
                <span
                  key={docId}
                  className="rounded border border-slate-300 bg-slate-100 px-1.5 py-0.5 font-mono text-[11px] text-slate-600"
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
