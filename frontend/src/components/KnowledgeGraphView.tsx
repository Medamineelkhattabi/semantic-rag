import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import type { GraphPath, GraphPayload } from '../types'
import { ENTITY_COLORS, entityColor } from './ui'

/** Left-to-right column order mirrors the supply chain the demo walks. */
export const COLUMN_ORDER = [
  'Employee',
  'Project',
  'Incident',
  'Component',
  'Supplier',
  'Contract',
  'Risk',
]

export const COLUMN_WIDTH = 232
export const ROW_HEIGHT = 74

// The entity node is a fixed size, so declare its dimensions up front rather
// than making React Flow measure the DOM first. Measurement costs a render
// round-trip (nodes sit `visibility: hidden` and edges do not draw until it
// completes), which shows up as a flash on load.
export const NODE_WIDTH = 172
export const NODE_HEIGHT = 62

type EntityNodeData = {
  label: string
  entityId: string
  type: string
  highlighted: boolean
  dimmed: boolean
  pathIndex: number | null
}

function EntityNode({ data }: NodeProps) {
  const d = data as unknown as EntityNodeData
  const color = entityColor(d.type)

  return (
    <div
      className={`relative rounded-lg border px-3 py-2 transition-all duration-300 ${color.bg} ${
        d.highlighted
          ? 'border-emerald-300/80 shadow-[0_0_22px_rgba(52,211,153,0.35)] scale-[1.04]'
          : `${color.border} ${d.dimmed ? 'opacity-25' : 'opacity-95'}`
      }`}
      style={{ width: NODE_WIDTH, height: NODE_HEIGHT }}
    >
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />

      {d.pathIndex !== null ? (
        <span className="absolute -left-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-emerald-400 text-[10px] font-bold text-emerald-950">
          {d.pathIndex + 1}
        </span>
      ) : null}

      <div className="flex items-center gap-1.5">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: color.dot }} />
        <span className="text-[8px] uppercase tracking-wider text-slate-500">{d.type}</span>
      </div>
      <div className={`truncate text-xs font-semibold ${color.text}`} title={d.entityId}>
        {d.entityId}
      </div>
      {d.label && d.label !== d.entityId ? (
        <div className="truncate text-[9px] text-slate-500" title={d.label}>
          {d.label}
        </div>
      ) : null}
    </div>
  )
}

const nodeTypes = { entity: EntityNode }

/* ------------------------------------------------------------------ */
export function buildLayout(payload: GraphPayload, path: GraphPath | null) {
  const pathNodeIds = new Map<string, number>()
  path?.nodes.forEach((node, index) => pathNodeIds.set(node.id, index))

  const pathEdgeKeys = new Set<string>()
  path?.steps.forEach((step) => {
    pathEdgeKeys.add(`${step.source}->${step.target}`)
    pathEdgeKeys.add(`${step.target}->${step.source}`)
  })

  const hasPath = pathNodeIds.size > 0

  // Group nodes into columns by entity type.
  const columns = new Map<string, GraphPayload['nodes']>()
  for (const node of payload.nodes) {
    const key = COLUMN_ORDER.includes(node.type) ? node.type : 'Other'
    if (!columns.has(key)) columns.set(key, [])
    columns.get(key)!.push(node)
  }

  const orderedKeys = [...COLUMN_ORDER, 'Other'].filter((key) => columns.has(key))
  const tallest = Math.max(...orderedKeys.map((key) => columns.get(key)!.length), 1)

  const nodes: Node[] = []
  orderedKeys.forEach((key, columnIndex) => {
    const group = columns
      .get(key)!
      .slice()
      .sort((a, b) => {
        // Path members float to the top of their column so the chain reads straight.
        const ai = pathNodeIds.has(a.id) ? -1 : 0
        const bi = pathNodeIds.has(b.id) ? -1 : 0
        if (ai !== bi) return ai - bi
        return a.id.localeCompare(b.id)
      })

    const offset = ((tallest - group.length) * ROW_HEIGHT) / 2
    group.forEach((node, rowIndex) => {
      nodes.push({
        id: node.id,
        type: 'entity',
        position: { x: columnIndex * COLUMN_WIDTH, y: offset + rowIndex * ROW_HEIGHT },
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
        data: {
          label: node.label,
          entityId: node.id,
          type: node.type,
          highlighted: pathNodeIds.has(node.id),
          dimmed: hasPath && !pathNodeIds.has(node.id),
          pathIndex: pathNodeIds.has(node.id) ? pathNodeIds.get(node.id)! : null,
        } satisfies EntityNodeData,
      })
    })
  })

  const edges: Edge[] = payload.edges.map((edge) => {
    const onPath = pathEdgeKeys.has(`${edge.source}->${edge.target}`)
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.type,
      animated: false,
      className: onPath ? 'edge-animated' : undefined,
      style: {
        stroke: onPath ? '#34d399' : '#2a3547',
        strokeWidth: onPath ? 2.4 : 1,
        opacity: hasPath && !onPath ? 0.2 : 1,
      },
      labelStyle: {
        fill: onPath ? '#6ee7b7' : '#64748b',
        fontSize: 9,
        fontFamily: 'ui-monospace, monospace',
      },
      labelBgStyle: { fill: '#0b0f18', fillOpacity: onPath ? 0.95 : 0.7 },
      labelBgPadding: [3, 1] as [number, number],
      labelBgBorderRadius: 3,
    }
  })

  return { nodes, edges }
}

/* ------------------------------------------------------------------ */
function GraphInner({
  payload,
  path,
}: {
  payload: GraphPayload
  path: GraphPath | null
}) {
  const layout = useMemo(() => buildLayout(payload, path), [payload, path])
  const [nodes, setNodes, onNodesChange] = useNodesState(layout.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(layout.edges)

  useEffect(() => {
    setNodes(layout.nodes)
    setEdges(layout.edges)
  }, [layout, setNodes, setEdges])

  const nodeColor = useCallback((node: Node) => {
    const type = (node.data as unknown as EntityNodeData)?.type ?? 'Unknown'
    return ENTITY_COLORS[type]?.dot ?? '#94a3b8'
  }, [])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.18 }}
      minZoom={0.15}
      maxZoom={2}
      proOptions={{ hideAttribution: true }}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1c2536" />
      <Controls
        className="!border-white/10 !bg-ink-850 [&>button]:!border-white/8 [&>button]:!bg-ink-800 [&>button]:!fill-slate-400"
        showInteractive={false}
      />
      <MiniMap
        pannable
        zoomable
        nodeColor={nodeColor}
        maskColor="rgba(7,9,15,0.75)"
        className="!border !border-white/8 !bg-ink-900"
      />
    </ReactFlow>
  )
}

export function KnowledgeGraphView({
  payload,
  path,
  height = 620,
}: {
  payload: GraphPayload
  path: GraphPath | null
  height?: number
}) {
  const [showLegend, setShowLegend] = useState(true)

  return (
    <div
      className="relative overflow-hidden rounded-xl border border-white/8 bg-ink-900/60"
      style={{ height }}
    >
      <ReactFlowProvider>
        <GraphInner payload={payload} path={path} />
      </ReactFlowProvider>

      {showLegend ? (
        <div className="absolute right-3 top-3 rounded-lg border border-white/8 bg-ink-850/95 px-3 py-2 backdrop-blur">
          <div className="mb-1.5 flex items-center justify-between gap-4">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              Entity types
            </span>
            <button
              onClick={() => setShowLegend(false)}
              className="text-[10px] text-slate-500 hover:text-slate-300"
            >
              hide
            </button>
          </div>
          <div className="flex flex-col gap-1">
            {COLUMN_ORDER.map((type) => (
              <div key={type} className="flex items-center gap-1.5">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ background: ENTITY_COLORS[type]?.dot }}
                />
                <span className="text-[10px] text-slate-400">{type}</span>
              </div>
            ))}
          </div>
          {path ? (
            <div className="mt-2 border-t border-white/8 pt-1.5">
              <div className="flex items-center gap-1.5">
                <span className="h-0.5 w-4 bg-emerald-400" />
                <span className="text-[10px] text-emerald-300">Answer path</span>
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <button
          onClick={() => setShowLegend(true)}
          className="absolute right-3 top-3 rounded-lg border border-white/8 bg-ink-850/95 px-2 py-1 text-[10px] text-slate-400 backdrop-blur hover:text-slate-200"
        >
          legend
        </button>
      )}
    </div>
  )
}
