import { Fragment, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AlertTriangle, FlaskConical, Play } from 'lucide-react'
import type { BenchmarkResponse } from '../types'
import { Card, EmptyState, ScoreBar, SectionTitle, Spinner, ms, pct } from './ui'

const METRICS = [
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'mrr', label: 'MRR' },
  { key: 'context_coverage', label: 'Context coverage' },
  { key: 'answer_correctness', label: 'Answer correctness' },
] as const

const AXIS = { stroke: '#64748b', fontSize: 12 }
const TOOLTIP_STYLE = {
  background: '#ffffff',
  border: '1px solid #cbd5e1',
  borderRadius: 8,
  fontSize: 13,
  color: '#0f172a',
  boxShadow: '0 8px 24px rgba(15,23,42,0.12)',
}

export function BenchmarkView({
  result,
  loading,
  onRun,
  disabled,
}: {
  result: BenchmarkResponse | null
  loading: boolean
  onRun: () => void
  disabled: boolean
}) {
  const [expanded, setExpanded] = useState<string | null>(null)

  const metricChartData = result
    ? METRICS.map((metric) => ({
        metric: metric.label,
        Basic: Number((result.summary.basic[metric.key] * 100).toFixed(1)),
        Semantic: Number((result.summary.semantic[metric.key] * 100).toFixed(1)),
      }))
    : []

  const hopChartData = result
    ? Object.keys(result.byHops.basic)
        .sort((a, b) => Number(a) - Number(b))
        .map((hop) => ({
          hops: `${hop}-hop`,
          Basic: Number((result.byHops.basic[hop].answer_correctness * 100).toFixed(1)),
          Semantic: Number((result.byHops.semantic[hop].answer_correctness * 100).toFixed(1)),
        }))
    : []

  const latencyData = result
    ? [
        {
          name: 'Retrieval',
          Basic: Number(result.summary.basic.retrieval_ms.toFixed(0)),
          Semantic: Number(result.summary.semantic.retrieval_ms.toFixed(0)),
        },
        {
          name: 'Generation',
          Basic: Number(result.summary.basic.generation_ms.toFixed(0)),
          Semantic: Number(result.summary.semantic.generation_ms.toFixed(0)),
        },
        {
          name: 'Total',
          Basic: Number(result.summary.basic.latency_ms.toFixed(0)),
          Semantic: Number(result.summary.semantic.latency_ms.toFixed(0)),
        },
      ]
    : []

  return (
    <div className="flex flex-col gap-4">
      <Card className="flex flex-wrap items-center justify-between gap-3 p-5">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-50 text-violet-700">
            <FlaskConical className="h-4 w-4" />
          </span>
          <div>
            <h2 className="text-[15px] font-semibold text-slate-900">Benchmark</h2>
            <p className="text-[13px] text-slate-600">
              Same questions, same ground truth, same LLM — every metric computed at run time.
            </p>
          </div>
        </div>
        <button
          onClick={onRun}
          disabled={loading || disabled}
          className="flex items-center gap-2 rounded-lg bg-gradient-to-br from-violet-500 to-sky-500 px-4 py-2 text-[15px] font-semibold text-white shadow-lg shadow-violet-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? <Spinner /> : <Play className="h-4 w-4" />}
          {loading ? 'Running…' : 'Run full benchmark'}
        </button>
      </Card>

      {loading ? (
        <Card className="p-10">
          <div className="flex flex-col items-center gap-3">
            <Spinner className="h-6 w-6 text-sky-600" />
            <p className="text-[15px] text-slate-800">Running both pipelines over every question…</p>
            <p className="text-[13px] text-slate-600">
              Two LLM calls per question — this takes a few minutes.
            </p>
          </div>
        </Card>
      ) : null}

      {!result && !loading ? (
        <EmptyState
          title="No benchmark run yet"
          body="Run the benchmark to score both pipelines on retrieval precision, recall, MRR, context coverage, answer correctness and latency."
        />
      ) : null}

      {result ? (
        <>
          {result.meta.errors.length ? (
            <Card className="border-rose-300 p-4">
              <div className="flex items-start gap-2">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
                <div>
                  <div className="text-[13px] font-semibold text-rose-700">
                    {result.meta.errors.length} question run(s) errored — excluded from the
                    averages rather than scored as zero
                  </div>
                  <ul className="mt-1 list-disc pl-4 text-[13px] text-slate-600">
                    {[...new Set(result.meta.errors)].slice(0, 5).map((error) => (
                      <li key={error}>{error}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </Card>
          ) : null}

          {/* Summary ------------------------------------------------ */}
          <Card className="p-5">
            <SectionTitle
              right={
                <span className="text-[12px] text-slate-600">
                  {result.summary.semantic.questions} scored
                  {result.summary.semantic.errored > 0
                    ? ` · ${result.summary.semantic.errored} excluded (errored)`
                    : ''}{' '}
                  · {ms(result.meta.wallClockMs)} wall clock
                </span>
              }
            >
              Overall comparison
            </SectionTitle>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-[15px]">
                <thead>
                  <tr className="border-b border-slate-200 text-[12px] uppercase tracking-wider text-slate-600">
                    <th className="py-2 text-left font-medium">Metric</th>
                    <th className="py-2 text-right font-medium text-amber-700">Basic RAG</th>
                    <th className="py-2 text-right font-medium text-emerald-700">Semantic RAG</th>
                    <th className="py-2 text-right font-medium">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map((metric) => {
                    const basic = result.summary.basic[metric.key]
                    const semantic = result.summary.semantic[metric.key]
                    const delta = semantic - basic
                    return (
                      <tr key={metric.key} className="border-b border-slate-200">
                        <td className="py-2.5 text-slate-800">{metric.label}</td>
                        <td className="py-2.5">
                          <div className="flex items-center justify-end gap-2">
                            <div className="w-20">
                              <ScoreBar value={basic} tone="basic" />
                            </div>
                            <span className="w-11 text-right font-mono tabular-nums text-slate-800">
                              {pct(basic)}
                            </span>
                          </div>
                        </td>
                        <td className="py-2.5">
                          <div className="flex items-center justify-end gap-2">
                            <div className="w-20">
                              <ScoreBar value={semantic} tone="semantic" />
                            </div>
                            <span className="w-11 text-right font-mono tabular-nums text-slate-800">
                              {pct(semantic)}
                            </span>
                          </div>
                        </td>
                        <td
                          className={`py-2.5 text-right font-mono text-[13px] tabular-nums ${
                            delta > 0.001
                              ? 'text-emerald-600'
                              : delta < -0.001
                                ? 'text-rose-600'
                                : 'text-slate-600'
                          }`}
                        >
                          {delta > 0 ? '+' : ''}
                          {(delta * 100).toFixed(1)}pp
                        </td>
                      </tr>
                    )
                  })}
                  <tr>
                    <td className="py-2.5 text-slate-800">Mean latency</td>
                    <td className="py-2.5 text-right font-mono tabular-nums text-slate-800">
                      {ms(result.summary.basic.latency_ms)}
                    </td>
                    <td className="py-2.5 text-right font-mono tabular-nums text-slate-800">
                      {ms(result.summary.semantic.latency_ms)}
                    </td>
                    <td
                      className={`py-2.5 text-right font-mono text-[13px] tabular-nums ${
                        result.summary.semantic.latency_ms > result.summary.basic.latency_ms
                          ? 'text-rose-600'
                          : 'text-emerald-600'
                      }`}
                    >
                      {result.summary.semantic.latency_ms > result.summary.basic.latency_ms
                        ? '+'
                        : ''}
                      {ms(
                        Math.abs(
                          result.summary.semantic.latency_ms - result.summary.basic.latency_ms,
                        ),
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>

          {/* Charts ------------------------------------------------- */}
          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="p-5">
              <SectionTitle>Retrieval &amp; answer quality (%)</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={metricChartData} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="metric" tick={AXIS} interval={0} angle={-14} textAnchor="end" height={62} />
                  <YAxis domain={[0, 100]} tick={AXIS} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(15,23,42,0.04)' }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Basic" fill="#d97706" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Semantic" fill="#059669" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card className="p-5">
              <SectionTitle>Answer correctness by hop count (%)</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={hopChartData} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="hops" tick={AXIS} />
                  <YAxis domain={[0, 100]} tick={AXIS} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="Basic"
                    stroke="#d97706"
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#d97706' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="Semantic"
                    stroke="#059669"
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#059669' }}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p className="mt-1 text-[12px] text-slate-600">
                The gap should widen as questions require more hops.
              </p>
            </Card>

            <Card className="p-5 lg:col-span-2">
              <SectionTitle>Mean latency breakdown (ms)</SectionTitle>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart
                  data={latencyData}
                  layout="vertical"
                  margin={{ top: 4, right: 16, bottom: 4, left: 26 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                  <XAxis type="number" tick={AXIS} />
                  <YAxis type="category" dataKey="name" tick={AXIS} width={70} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(15,23,42,0.04)' }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Basic" fill="#d97706" radius={[0, 3, 3, 0]} />
                  <Bar dataKey="Semantic" fill="#059669" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Per-question table ------------------------------------- */}
          <Card className="p-5">
            <SectionTitle>Per-question results</SectionTitle>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[900px] text-[13px]">
                <thead>
                  <tr className="border-b border-slate-200 text-[12px] uppercase tracking-wider text-slate-600">
                    <th className="py-2 pr-2 text-left font-medium">Question</th>
                    <th className="px-1 py-2 text-center font-medium">Hops</th>
                    <th className="px-2 py-2 text-center font-medium text-amber-700" colSpan={4}>
                      Basic
                    </th>
                    <th className="px-2 py-2 text-center font-medium text-emerald-700" colSpan={4}>
                      Semantic
                    </th>
                  </tr>
                  <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wider text-slate-600">
                    <th /> <th />
                    <th className="px-1 py-1 text-right">P</th>
                    <th className="px-1 py-1 text-right">R</th>
                    <th className="px-1 py-1 text-right">MRR</th>
                    <th className="px-1 py-1 text-right">Ans</th>
                    <th className="px-1 py-1 text-right">P</th>
                    <th className="px-1 py-1 text-right">R</th>
                    <th className="px-1 py-1 text-right">MRR</th>
                    <th className="px-1 py-1 text-right">Ans</th>
                  </tr>
                </thead>
                <tbody>
                  {result.questions.map((row) => {
                    const open = expanded === row.id
                    const better = row.semantic.answer_correctness > row.basic.answer_correctness
                    return (
                      <Fragment key={row.id}>
                        <tr
                          onClick={() => setExpanded(open ? null : row.id)}
                          className={`cursor-pointer border-b border-slate-200 transition hover:bg-slate-100 ${
                            open ? 'bg-slate-100' : ''
                          }`}
                        >
                          <td className="max-w-[380px] py-2 pr-2">
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-[12px] text-slate-600">{row.id}</span>
                              {better ? <span className="text-emerald-600">▲</span> : null}
                              <span className="truncate text-slate-800">{row.question}</span>
                            </div>
                          </td>
                          <td className="px-1 py-2 text-center font-mono text-slate-600">
                            {row.hops}
                          </td>
                          {[row.basic, row.semantic].map((score, index) => (
                            <Fragment key={index}>
                              <td
                                className="px-1 py-2 text-right font-mono tabular-nums text-slate-600"
                              >
                                {pct(score.precision)}
                              </td>
                              <td
                                className="px-1 py-2 text-right font-mono tabular-nums text-slate-600"
                              >
                                {pct(score.recall)}
                              </td>
                              <td
                                className="px-1 py-2 text-right font-mono tabular-nums text-slate-600"
                              >
                                {score.mrr.toFixed(2)}
                              </td>
                              <td
                                className={`px-1 py-2 text-right font-mono tabular-nums ${
                                  score.answer_correctness >= 0.999
                                    ? 'text-emerald-600'
                                    : score.answer_correctness <= 0.001
                                      ? 'text-rose-600'
                                      : 'text-slate-800'
                                }`}
                              >
                                {pct(score.answer_correctness)}
                              </td>
                            </Fragment>
                          ))}
                        </tr>
                        {open ? (
                          <tr className="border-b border-slate-200 bg-slate-50">
                            <td colSpan={10} className="px-3 py-3">
                              <div className="grid gap-3 md:grid-cols-2">
                                {(
                                  [
                                    ['Basic RAG', row.basic, 'amber'],
                                    ['Semantic RAG', row.semantic, 'emerald'],
                                  ] as const
                                ).map(([label, score, tone]) => (
                                  <div
                                    key={label}
                                    className={`rounded-lg border p-3 ${
                                      tone === 'amber'
                                        ? 'border-amber-300'
                                        : 'border-emerald-300'
                                    }`}
                                  >
                                    <div
                                      className={`mb-1.5 text-[12px] font-semibold uppercase tracking-wider ${
                                        tone === 'amber' ? 'text-amber-700' : 'text-emerald-700'
                                      }`}
                                    >
                                      {label}
                                    </div>
                                    <p className="mb-2 max-h-32 overflow-y-auto whitespace-pre-wrap text-[13px] leading-relaxed text-slate-600">
                                      {score.answer || score.error || '—'}
                                    </p>
                                    <div className="flex flex-wrap gap-1">
                                      {score.retrieved_docs.map((doc) => (
                                        <span
                                          key={doc}
                                          className={`rounded px-1 py-0.5 font-mono text-[11px] ${
                                            row.relevantDocs.includes(doc)
                                              ? 'bg-emerald-50 text-emerald-700'
                                              : 'bg-slate-100 text-slate-600'
                                          }`}
                                        >
                                          {doc}
                                        </span>
                                      ))}
                                    </div>
                                    {score.missed_keypoints.length ? (
                                      <div className="mt-1.5 text-[12px] text-rose-600">
                                        missed: {score.missed_keypoints.join(', ')}
                                      </div>
                                    ) : null}
                                  </div>
                                ))}
                              </div>
                              <div className="mt-2 text-[12px] text-slate-600">
                                Ground-truth documents: {row.relevantDocs.join(', ')}
                              </div>
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* By difficulty ------------------------------------------ */}
          <Card className="p-5">
            <SectionTitle>By difficulty</SectionTitle>
            <div className="grid gap-3 sm:grid-cols-3">
              {Object.keys(result.byDifficulty.basic).map((level) => {
                const basic = result.byDifficulty.basic[level]
                const semantic = result.byDifficulty.semantic[level]
                return (
                  <div key={level} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-[13px] font-semibold capitalize text-slate-800">{level}</span>
                      <span className="text-[12px] text-slate-600">{basic.questions}q</span>
                    </div>
                    {(
                      [
                        ['Answer', 'answer_correctness'],
                        ['Recall', 'recall'],
                        ['Coverage', 'context_coverage'],
                      ] as const
                    ).map(([label, key]) => (
                      <div key={key} className="mb-1.5">
                        <div className="flex justify-between text-[12px] text-slate-600">
                          <span>{label}</span>
                          <span className="font-mono">
                            <span className="text-amber-700">{pct(basic[key])}</span>
                            <span className="mx-1 text-slate-9000">/</span>
                            <span className="text-emerald-700">{pct(semantic[key])}</span>
                          </span>
                        </div>
                        <div className="mt-0.5 flex gap-1">
                          <div className="flex-1">
                            <ScoreBar value={basic[key]} tone="basic" />
                          </div>
                          <div className="flex-1">
                            <ScoreBar value={semantic[key]} tone="semantic" />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>
          </Card>
        </>
      ) : null}
    </div>
  )
}
