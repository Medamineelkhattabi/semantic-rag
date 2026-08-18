import type {
  AppConfig,
  BenchmarkQuestion,
  BenchmarkResponse,
  CompareResponse,
  DatasetDocument,
  EngineStatus,
  GraphPayload,
} from '../types'

const BASE = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail)
  }

  return response.json() as Promise<T>
}

export const api = {
  status: () => request<EngineStatus>('/api/status'),
  config: () => request<AppConfig>('/api/config'),
  build: () => request<EngineStatus>('/api/build', { method: 'POST' }),

  dataset: () =>
    request<{ count: number; documents: DatasetDocument[] }>('/api/dataset'),

  document: (docId: string) =>
    request<{ docId: string; title: string; text: string }>(`/api/dataset/${docId}`),

  questions: () =>
    request<{ count: number; questions: BenchmarkQuestion[] }>('/api/benchmark/questions'),

  compare: (question: string) =>
    request<CompareResponse>('/api/compare', {
      method: 'POST',
      body: JSON.stringify({ question }),
    }),

  graph: () => request<GraphPayload>('/api/graph'),

  runBenchmark: (questionIds?: string[]) =>
    request<BenchmarkResponse>('/api/benchmark/run', {
      method: 'POST',
      body: JSON.stringify({ questionIds: questionIds ?? null }),
    }),
}
