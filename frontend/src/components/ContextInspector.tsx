import { useState } from 'react'
import { Check, Copy, Terminal } from 'lucide-react'
import type { BasicResult, SemanticResult } from '../types'
import { Card, SectionTitle } from './ui'

function ContextPane({
  title,
  tone,
  context,
  meta,
}: {
  title: string
  tone: 'basic' | 'semantic'
  context: string
  meta: { label: string; value: string }[]
}) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(context)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* clipboard unavailable (insecure origin) — ignore */
    }
  }

  const accent = tone === 'basic' ? 'text-amber-700' : 'text-emerald-700'
  const border = tone === 'basic' ? 'border-amber-300' : 'border-emerald-300'

  return (
    <div className={`flex min-h-0 flex-col rounded-lg border ${border} bg-slate-50`}>
      <div className="flex items-center justify-between gap-2 border-b border-slate-200 px-3 py-2">
        <span className={`text-[13px] font-semibold uppercase tracking-wider ${accent}`}>
          {title}
        </span>
        <button
          onClick={copy}
          className="flex items-center gap-1 rounded border border-slate-200 px-1.5 py-0.5 text-[12px] text-slate-600 transition hover:bg-slate-100 hover:text-slate-800"
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? 'copied' : 'copy'}
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5 border-b border-slate-200 px-3 py-2">
        {meta.map((item) => (
          <span
            key={item.label}
            className="rounded border border-slate-200 bg-slate-100 px-1.5 py-0.5 text-[12px] text-slate-600"
          >
            {item.label} <span className="font-medium tabular-nums text-slate-800">{item.value}</span>
          </span>
        ))}
      </div>

      <pre className="max-h-[460px] overflow-auto whitespace-pre-wrap break-words px-3 py-3 font-mono text-[12px] leading-relaxed text-slate-600">
        {context || '—'}
      </pre>
    </div>
  )
}

/**
 * Shows the exact prompt context each pipeline handed to the LLM. This is the
 * honest core of the comparison: same model, same question, different context.
 */
export function ContextInspector({
  basic,
  semantic,
}: {
  basic: BasicResult
  semantic: SemanticResult
}) {
  return (
    <Card className="p-5">
      <SectionTitle
        icon={<Terminal className="h-3.5 w-3.5" />}
        right={
          <span className="text-[12px] text-slate-600">
            Identical question and model — only the context differs
          </span>
        }
      >
        Context Inspector — exactly what each pipeline sent to the LLM
      </SectionTitle>

      <div className="grid gap-4 lg:grid-cols-2">
        <ContextPane
          title="Basic RAG context"
          tone="basic"
          context={basic.context}
          meta={[
            { label: 'chars', value: basic.context.length.toLocaleString() },
            { label: 'chunks', value: String(basic.chunks.length) },
            { label: 'docs', value: String(basic.sources.length) },
            {
              label: 'prompt tokens',
              value: String((basic.stats.prompt_tokens as number) ?? '—'),
            },
          ]}
        />
        <ContextPane
          title="Semantic RAG context"
          tone="semantic"
          context={semantic.context}
          meta={[
            { label: 'chars', value: semantic.context.length.toLocaleString() },
            { label: 'facts', value: String(semantic.relationships.length) },
            { label: 'passages', value: String(semantic.passages.length) },
            {
              label: 'prompt tokens',
              value: String((semantic.stats.prompt_tokens as number) ?? '—'),
            },
          ]}
        />
      </div>
    </Card>
  )
}
