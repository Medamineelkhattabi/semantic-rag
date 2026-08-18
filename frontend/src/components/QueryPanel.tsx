import { useState } from 'react'
import { Play, Search, Zap } from 'lucide-react'
import type { BenchmarkQuestion } from '../types'
import { Card, Spinner } from './ui'

const DIFFICULTY_TONE: Record<string, string> = {
  easy: 'border-sky-400/25 bg-sky-500/8 text-sky-300 hover:bg-sky-500/15',
  medium: 'border-amber-400/25 bg-amber-500/8 text-amber-300 hover:bg-amber-500/15',
  hard: 'border-rose-400/25 bg-rose-500/8 text-rose-300 hover:bg-rose-500/15',
}

export function QueryPanel({
  questions,
  onCompare,
  loading,
  disabled,
}: {
  questions: BenchmarkQuestion[]
  onCompare: (question: string) => void
  loading: boolean
  disabled: boolean
}) {
  const [value, setValue] = useState('')

  const submit = () => {
    const trimmed = value.trim()
    if (trimmed.length >= 3 && !loading && !disabled) onCompare(trimmed)
  }

  return (
    <Card className="p-5">
      <div className="flex items-start gap-3">
        <Search className="mt-2.5 h-4 w-4 shrink-0 text-slate-500" />
        <textarea
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) submit()
          }}
          rows={2}
          placeholder="Ask one question — it runs through both pipelines on the same corpus and the same LLM…"
          className="min-h-[52px] flex-1 resize-y rounded-lg border border-white/8 bg-black/25 px-3 py-2.5 text-sm text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-sky-400/40 focus:ring-1 focus:ring-sky-400/20"
        />
        <button
          onClick={submit}
          disabled={loading || disabled || value.trim().length < 3}
          className="flex h-[52px] shrink-0 items-center gap-2 rounded-lg bg-gradient-to-br from-sky-500 to-emerald-500 px-5 text-sm font-semibold text-white shadow-lg shadow-sky-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
        >
          {loading ? <Spinner /> : <Zap className="h-4 w-4" />}
          Compare Both
        </button>
      </div>

      <div className="mt-4">
        <div className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500">
          <Play className="h-3 w-3" />
          Benchmark questions — click to load
        </div>
        <div className="flex flex-wrap gap-1.5">
          {questions.map((question) => (
            <button
              key={question.id}
              onClick={() => setValue(question.question)}
              title={question.note || question.question}
              className={`group max-w-full rounded-md border px-2 py-1 text-left text-[11px] transition ${
                DIFFICULTY_TONE[question.difficulty] ?? DIFFICULTY_TONE.easy
              }`}
            >
              <span className="mr-1 font-mono opacity-60">{question.id}</span>
              <span className="opacity-50">{question.hops}-hop</span>
              <span className="mx-1.5 opacity-30">|</span>
              <span className="text-slate-300 group-hover:text-slate-100">
                {question.question.length > 88
                  ? `${question.question.slice(0, 88)}…`
                  : question.question}
              </span>
            </button>
          ))}
        </div>
      </div>
    </Card>
  )
}
