import { useEffect, useState } from 'react'
import { FileText, Library } from 'lucide-react'
import { api } from '../lib/api'
import type { DatasetDocument } from '../types'
import { Card, SectionTitle, Spinner } from './ui'

export function DatasetView() {
  const [documents, setDocuments] = useState<DatasetDocument[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api
      .dataset()
      .then((payload) => {
        setDocuments(payload.documents)
        if (payload.documents.length) setSelected(payload.documents[0].docId)
      })
      .catch(() => setDocuments([]))
  }, [])

  useEffect(() => {
    if (!selected) return
    setLoading(true)
    api
      .document(selected)
      .then((payload) => setText(payload.text))
      .catch((error: Error) => setText(`Failed to load: ${error.message}`))
      .finally(() => setLoading(false))
  }, [selected])

  return (
    <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
      <Card className="p-4">
        <SectionTitle icon={<Library className="h-3.5 w-3.5" />}>
          Corpus ({documents.length})
        </SectionTitle>
        <div className="flex max-h-[640px] flex-col gap-1 overflow-y-auto">
          {documents.map((doc) => (
            <button
              key={doc.docId}
              onClick={() => setSelected(doc.docId)}
              className={`rounded-lg border px-2.5 py-2 text-left transition ${
                selected === doc.docId
                  ? 'border-sky-400/35 bg-sky-500/10'
                  : 'border-transparent hover:bg-white/[0.03]'
              }`}
            >
              <div className="truncate text-[11px] font-medium text-slate-200">{doc.title}</div>
              <div className="mt-0.5 flex items-center gap-2 text-[9px] text-slate-500">
                <span className="font-mono">{doc.docId}</span>
                <span>{(doc.chars / 1000).toFixed(1)}k chars</span>
              </div>
            </button>
          ))}
        </div>
        <p className="mt-3 border-t border-white/6 pt-3 text-[10px] leading-relaxed text-slate-500">
          Both pipelines ingest this exact corpus. Facts in the supply chain are deliberately split
          across documents so that no single chunk contains two consecutive links.
        </p>
      </Card>

      <Card className="flex min-h-[400px] flex-col p-4">
        <SectionTitle icon={<FileText className="h-3.5 w-3.5" />}>
          {selected ?? 'Select a document'}
        </SectionTitle>
        {loading ? (
          <div className="flex flex-1 items-center justify-center">
            <Spinner className="h-5 w-5 text-sky-400" />
          </div>
        ) : (
          <pre className="max-h-[640px] flex-1 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-black/25 px-4 py-3 font-mono text-[11px] leading-relaxed text-slate-400">
            {text}
          </pre>
        )}
      </Card>
    </div>
  )
}
