import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  BrainCircuit,
  Database,
  FlaskConical,
  GitBranch,
  Network,
} from 'lucide-react'
import { api } from './lib/api'
import type {
  AppConfig,
  BenchmarkQuestion,
  BenchmarkResponse,
  CompareResponse,
  EngineStatus,
  GraphPayload,
} from './types'
import { BasicColumn, SemanticColumn } from './components/ComparisonColumns'
import { BenchmarkView } from './components/BenchmarkView'
import { ContextInspector } from './components/ContextInspector'
import { DatasetView } from './components/DatasetView'
import { KnowledgeGraphView } from './components/KnowledgeGraphView'
import { QueryPanel } from './components/QueryPanel'
import { WhySemanticRag } from './components/WhySemanticRag'
import { Card, EmptyState, Spinner } from './components/ui'

type Tab = 'compare' | 'graph' | 'benchmark' | 'dataset'

const TABS: { id: Tab; label: string; icon: typeof Network }[] = [
  { id: 'compare', label: 'Compare', icon: GitBranch },
  { id: 'graph', label: 'Knowledge Graph', icon: Network },
  { id: 'benchmark', label: 'Benchmark', icon: FlaskConical },
  { id: 'dataset', label: 'Dataset', icon: Database },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('compare')
  const [status, setStatus] = useState<EngineStatus | null>(null)
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [questions, setQuestions] = useState<BenchmarkQuestion[]>([])

  const [comparison, setComparison] = useState<CompareResponse | null>(null)
  const [comparing, setComparing] = useState(false)
  const [error, setError] = useState('')

  const [graph, setGraph] = useState<GraphPayload | null>(null)
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null)
  const [benchmarkLoading, setBenchmarkLoading] = useState(false)

  const ready = status?.state === 'ready'

  /* Poll engine status until the pipelines finish building. */
  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    const poll = async () => {
      try {
        const payload = await api.status()
        if (cancelled) return
        setStatus(payload)
        if (payload.state !== 'ready' && payload.state !== 'error') {
          timer = window.setTimeout(poll, 1500)
        }
      } catch {
        if (!cancelled) timer = window.setTimeout(poll, 2500)
      }
    }

    poll()
    api.config().then(setConfig).catch(() => undefined)
    api.questions().then((p) => setQuestions(p.questions)).catch(() => undefined)

    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [])

  /* Load the graph once the engine is ready. */
  useEffect(() => {
    if (!ready || graph) return
    api.graph().then(setGraph).catch(() => undefined)
  }, [ready, graph])

  const runCompare = useCallback(async (question: string) => {
    setComparing(true)
    setError('')
    try {
      const payload = await api.compare(question)
      setComparison(payload)
    } catch (exc) {
      setError((exc as Error).message)
    } finally {
      setComparing(false)
    }
  }, [])

  const runBenchmark = useCallback(async () => {
    setBenchmarkLoading(true)
    setError('')
    try {
      setBenchmark(await api.runBenchmark())
    } catch (exc) {
      setError((exc as Error).message)
    } finally {
      setBenchmarkLoading(false)
    }
  }, [])

  const primaryPath = comparison?.semantic.primaryPath ?? null

  const graphStats = useMemo(() => {
    if (!graph) return null
    return graph.stats as { entities?: number; relationships?: number }
  }, [graph])

  return (
    <div className="min-h-screen">
      {/* Header ---------------------------------------------------- */}
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/85 backdrop-blur-xl shadow-sm shadow-slate-900/5">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-3 px-6 py-3">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-sky-500 to-emerald-500 shadow-lg shadow-sky-500/25">
              <BrainCircuit className="h-5 w-5 text-white" />
            </span>
            <div>
              <h1 className="text-[17px] font-semibold tracking-tight text-slate-900">
                RAG Intelligence Lab
              </h1>
              <p className="text-[13px] text-slate-600">
                Basic RAG vs Semantic RAG · powered by Semantica
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {config ? (
              <>
                <Chip label="LLM" value={config.llmModel} />
                <Chip label="Embeddings" value={config.embeddingModel} />
                <Chip label="top-K" value={String(config.topK)} />
                <Chip label="max hops" value={String(config.graphMaxHops)} />
              </>
            ) : null}
            <StatusPill status={status} />
          </div>
        </div>

        <div className="mx-auto flex max-w-[1600px] gap-1 px-6">
          {TABS.map((item) => {
            const Icon = item.icon
            const active = tab === item.id
            return (
              <button
                key={item.id}
                onClick={() => setTab(item.id)}
                className={`flex items-center gap-1.5 border-b-2 px-3 py-2 text-[13px] font-medium transition ${
                  active
                    ? 'border-sky-600 text-slate-900'
                    : 'border-transparent text-slate-600 hover:text-slate-800'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {item.label}
              </button>
            )
          })}
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] px-6 py-5">
        {status?.state === 'building' ? (
          <Card className="mb-4 border-sky-400/25 p-4">
            <div className="flex items-center gap-3">
              <Spinner className="h-4 w-4 text-sky-600" />
              <div>
                <div className="text-[13px] font-medium text-slate-800">
                  Building pipelines — {status.stage || 'starting'}
                </div>
                <div className="text-[13px] text-slate-600">
                  The knowledge graph is extracted once with the LLM, then cached to disk.
                </div>
              </div>
            </div>
          </Card>
        ) : null}

        {status?.state === 'error' ? (
          <Card className="mb-4 border-rose-400/30 p-4">
            <div className="flex items-start gap-2">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
              <div>
                <div className="text-[13px] font-semibold text-rose-700">Engine build failed</div>
                <div className="mt-0.5 font-mono text-[13px] text-slate-600">{status.error}</div>
                <button
                  onClick={() => api.build().then(setStatus).catch(() => undefined)}
                  className="mt-2 rounded border border-slate-300 px-2 py-1 text-[13px] text-slate-800 hover:bg-slate-100"
                >
                  Retry build
                </button>
              </div>
            </div>
          </Card>
        ) : null}

        {error ? (
          <Card className="mb-4 border-rose-400/30 p-3">
            <div className="flex items-center gap-2 text-[13px] text-rose-700">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          </Card>
        ) : null}

        {/* Compare ------------------------------------------------- */}
        {tab === 'compare' ? (
          <div className="flex flex-col gap-4">
            <QueryPanel
              questions={questions}
              onCompare={runCompare}
              loading={comparing}
              disabled={!ready}
            />

            {comparing ? (
              <Card className="p-10">
                <div className="flex flex-col items-center gap-3">
                  <Spinner className="h-6 w-6 text-sky-600" />
                  <p className="text-[15px] text-slate-800">
                    Running both pipelines on the same question…
                  </p>
                </div>
              </Card>
            ) : null}

            {comparison && !comparing ? (
              <>
                <WhySemanticRag
                  path={primaryPath}
                  basicSources={comparison.basic.sources}
                  semanticSources={comparison.semantic.sources}
                />

                <div className="grid gap-4 lg:grid-cols-2">
                  <BasicColumn result={comparison.basic} />
                  <SemanticColumn result={comparison.semantic} />
                </div>

                {graph ? (
                  <Card className="p-5">
                    <div className="mb-3 flex items-center justify-between">
                      <div className="flex items-center gap-2 text-[13px] font-semibold uppercase tracking-[0.13em] text-slate-600">
                        <Network className="h-3.5 w-3.5" />
                        Knowledge graph — path used for this answer
                      </div>
                      <button
                        onClick={() => setTab('graph')}
                        className="text-[13px] text-sky-600 hover:text-sky-700"
                      >
                        open full view →
                      </button>
                    </div>
                    <KnowledgeGraphView payload={graph} path={primaryPath} height={430} />
                  </Card>
                ) : null}

                <ContextInspector basic={comparison.basic} semantic={comparison.semantic} />
              </>
            ) : null}

            {!comparison && !comparing ? (
              <EmptyState
                title={ready ? 'Ask a question to begin' : 'Waiting for the engine to build…'}
                body="Both pipelines run on the same corpus with the same LLM and embedding model. Only the retrieval strategy differs."
              />
            ) : null}
          </div>
        ) : null}

        {/* Graph --------------------------------------------------- */}
        {tab === 'graph' ? (
          <Card className="p-5">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-[15px] font-semibold text-slate-900">Knowledge Graph</h2>
                <p className="text-[13px] text-slate-600">
                  Extracted from the corpus by Semantica.
                  {graphStats
                    ? ` ${graphStats.entities ?? 0} entities · ${graphStats.relationships ?? 0} relationships.`
                    : ''}
                  {primaryPath ? ' The highlighted path answered your last question.' : ''}
                </p>
              </div>
              {primaryPath ? (
                <span className="rounded-lg border border-emerald-400/25 bg-emerald-500/8 px-2.5 py-1 font-mono text-[13px] text-emerald-700">
                  {primaryPath.chain}
                </span>
              ) : null}
            </div>
            {graph ? (
              <KnowledgeGraphView payload={graph} path={primaryPath} height={680} />
            ) : (
              <EmptyState
                title="Graph not built yet"
                body="The knowledge graph appears once extraction finishes."
              />
            )}
          </Card>
        ) : null}

        {/* Benchmark ----------------------------------------------- */}
        {tab === 'benchmark' ? (
          <BenchmarkView
            result={benchmark}
            loading={benchmarkLoading}
            onRun={runBenchmark}
            disabled={!ready}
          />
        ) : null}

        {/* Dataset ------------------------------------------------- */}
        {tab === 'dataset' ? <DatasetView /> : null}
      </main>
    </div>
  )
}

/* ------------------------------------------------------------------ */
function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="hidden items-center gap-1 rounded-md border border-slate-200 bg-slate-100 px-2 py-1 text-[12px] text-slate-600 md:inline-flex">
      {label}
      <span className="font-mono text-slate-800">{value}</span>
    </span>
  )
}

function StatusPill({ status }: { status: EngineStatus | null }) {
  const state = status?.state ?? 'idle'
  const map = {
    ready: { text: 'ready', color: 'bg-emerald-400', label: 'text-emerald-700' },
    building: { text: 'building', color: 'bg-amber-400', label: 'text-amber-700' },
    error: { text: 'error', color: 'bg-rose-400', label: 'text-rose-700' },
    idle: { text: 'idle', color: 'bg-slate-400', label: 'text-slate-600' },
  }[state]

  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-100 px-2 py-1 text-[12px]">
      <span className={`h-1.5 w-1.5 rounded-full ${map.color} ${state === 'building' ? 'animate-pulse' : ''}`} />
      <span className={map.label}>{map.text}</span>
    </span>
  )
}
