export interface Latency {
  total_ms: number
  retrieval_ms: number
  generation_ms: number
  entity_linking_ms?: number
  traversal_ms?: number
  context_expansion_ms?: number
}

export interface RetrievedChunk {
  chunkId: string
  docId: string
  docTitle: string
  text: string
  score: number
  rank: number
}

export interface BasicResult {
  answer: string
  chunks: RetrievedChunk[]
  sources: string[]
  rankedDocs: string[]
  context: string
  latency: Latency
  stats: Record<string, unknown>
}

export interface SeedEntity {
  id: string
  label: string
  type: string
  score: number
  method: 'identifier' | 'alias' | 'embedding'
}

export interface Relationship {
  source: string
  relation: string
  target: string
  docIds: string[]
  evidence: string
  onPath: boolean
  sentence: string
}

export interface PathNode {
  id: string
  label: string
  type: string
}

export interface PathStep {
  source: string
  relation: string
  target: string
  direction: 'forward' | 'reverse'
  docIds: string[]
  evidence: string
  phrase: string
}

export interface GraphPath {
  nodes: PathNode[]
  steps: PathStep[]
  hops: number
  chain: string
}

export interface Passage {
  docId: string
  docTitle: string
  heading: string
  text: string
  matchedEntities: string[]
}

export interface SemanticResult {
  answer: string
  entities: SeedEntity[]
  targetTypes: string[]
  relationships: Relationship[]
  paths: GraphPath[]
  primaryPath: GraphPath | null
  passages: Passage[]
  subgraphNodes: string[]
  sources: string[]
  rankedDocs: string[]
  context: string
  latency: Latency
  stats: Record<string, unknown>
}

export interface CompareResponse {
  question: string
  basic: BasicResult
  semantic: SemanticResult
  shared: {
    llmModel: string
    embeddingModel: string
    documents: number
    temperature: number
  }
}

export interface GraphNodePayload {
  id: string
  label: string
  type: string
  docIds: string[]
  attributes: Record<string, unknown>
}

export interface GraphEdgePayload {
  id: string
  source: string
  target: string
  type: string
  docIds: string[]
}

export interface GraphPayload {
  nodes: GraphNodePayload[]
  edges: GraphEdgePayload[]
  stats: Record<string, unknown>
}

export interface BenchmarkQuestion {
  id: string
  question: string
  hops: number
  difficulty: 'easy' | 'medium' | 'hard'
  relevantDocs: string[]
  goldPath: string[]
  note: string
}

export interface QuestionScore {
  question_id: string
  question: string
  pipeline: string
  hops: number
  difficulty: string
  precision: number
  recall: number
  f1: number
  mrr: number
  context_coverage: number
  answer_correctness: number
  latency_ms: number
  retrieval_ms: number
  generation_ms: number
  retrieved_docs: string[]
  relevant_docs: string[]
  missed_keypoints: string[]
  answer: string
  error: string
}

export interface MetricSummary {
  precision: number
  recall: number
  f1: number
  mrr: number
  context_coverage: number
  answer_correctness: number
  latency_ms: number
  retrieval_ms: number
  generation_ms: number
  questions: number
  /** Questions excluded from the mean because they errored (e.g. gateway 503). */
  errored: number
}

export interface BenchmarkResponse {
  questions: {
    id: string
    question: string
    hops: number
    difficulty: string
    relevantDocs: string[]
    basic: QuestionScore
    semantic: QuestionScore
  }[]
  summary: { basic: MetricSummary; semantic: MetricSummary }
  byDifficulty: Record<string, Record<string, MetricSummary>>
  byHops: Record<string, Record<string, MetricSummary>>
  meta: { questionCount: number; wallClockMs: number; errors: string[] }
}

export interface EngineStatus {
  state: 'idle' | 'building' | 'ready' | 'error'
  stage: string
  error: string
  builtAt: number | null
  basic: Record<string, unknown>
  semantic: Record<string, unknown>
}

export interface DatasetDocument {
  docId: string
  title: string
  path: string
  chars: number
  metadata: Record<string, string>
}

export interface AppConfig {
  llmModel: string
  extractionModel: string
  embeddingModel: string
  temperature: number
  chunkSize: number
  chunkOverlap: number
  topK: number
  graphMaxHops: number
  baseUrl: string
}
