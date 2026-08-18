import { describe, expect, it } from 'vitest'
import {
  buildLayout,
  COLUMN_ORDER,
  COLUMN_WIDTH,
  NODE_HEIGHT,
  NODE_WIDTH,
  ROW_HEIGHT,
} from './KnowledgeGraphView'
import type { GraphPath, GraphPayload } from '../types'

/**
 * These cover the pure layout/highlight logic. React Flow's own SVG drawing is
 * driven by requestAnimationFrame and a ResizeObserver, so it cannot be
 * exercised in a non-compositing environment — but everything we hand it can.
 */

const payload: GraphPayload = {
  nodes: [
    { id: 'Project Phoenix', label: 'Project Phoenix', type: 'Project', docIds: ['d1'], attributes: {} },
    { id: 'Project Titan', label: 'Project Titan', type: 'Project', docIds: ['d1'], attributes: {} },
    { id: 'C-17', label: 'C-17', type: 'Component', docIds: ['d2'], attributes: {} },
    { id: 'C-22', label: 'C-22', type: 'Component', docIds: ['d2'], attributes: {} },
    { id: 'Alpha Precision Systems', label: 'Alpha Precision Systems', type: 'Supplier', docIds: ['d3'], attributes: {} },
    { id: 'C-2048', label: 'C-2048', type: 'Contract', docIds: ['d4'], attributes: {} },
    { id: 'R-17', label: 'R-17', type: 'Risk', docIds: ['d5'], attributes: {} },
    { id: 'Yuki Tanaka', label: 'Yuki Tanaka', type: 'Employee', docIds: ['d6'], attributes: {} },
  ],
  edges: [
    { id: 'e1', source: 'Project Phoenix', target: 'C-17', type: 'uses', docIds: ['d1'] },
    { id: 'e2', source: 'C-17', target: 'Alpha Precision Systems', type: 'supplied_by', docIds: ['d2'] },
    { id: 'e3', source: 'Alpha Precision Systems', target: 'C-2048', type: 'governed_by', docIds: ['d3'] },
    { id: 'e4', source: 'C-2048', target: 'R-17', type: 'has_risk', docIds: ['d4'] },
    { id: 'e5', source: 'Project Titan', target: 'C-22', type: 'uses', docIds: ['d1'] },
  ],
  stats: {},
}

const path: GraphPath = {
  nodes: [
    { id: 'Project Phoenix', label: 'Project Phoenix', type: 'Project' },
    { id: 'C-17', label: 'C-17', type: 'Component' },
    { id: 'Alpha Precision Systems', label: 'Alpha Precision Systems', type: 'Supplier' },
    { id: 'C-2048', label: 'C-2048', type: 'Contract' },
    { id: 'R-17', label: 'R-17', type: 'Risk' },
  ],
  steps: [
    { source: 'Project Phoenix', relation: 'uses', target: 'C-17', direction: 'forward', docIds: ['d1'], evidence: '', phrase: '' },
    { source: 'C-17', relation: 'supplied_by', target: 'Alpha Precision Systems', direction: 'forward', docIds: ['d2'], evidence: '', phrase: '' },
    { source: 'Alpha Precision Systems', relation: 'governed_by', target: 'C-2048', direction: 'forward', docIds: ['d3'], evidence: '', phrase: '' },
    { source: 'C-2048', relation: 'has_risk', target: 'R-17', direction: 'forward', docIds: ['d4'], evidence: '', phrase: '' },
  ],
  hops: 4,
  chain: 'Project Phoenix  ->  C-17  ->  Alpha Precision Systems  ->  C-2048  ->  R-17',
}

const byId = (nodes: ReturnType<typeof buildLayout>['nodes']) =>
  Object.fromEntries(nodes.map((n) => [n.id, n]))

describe('buildLayout — structure', () => {
  it('emits one node per entity and one edge per relationship', () => {
    const { nodes, edges } = buildLayout(payload, null)
    expect(nodes).toHaveLength(payload.nodes.length)
    expect(edges).toHaveLength(payload.edges.length)
  })

  it('preserves every edge endpoint', () => {
    const { nodes, edges } = buildLayout(payload, null)
    const ids = new Set(nodes.map((n) => n.id))
    for (const edge of edges) {
      expect(ids.has(edge.source)).toBe(true)
      expect(ids.has(edge.target)).toBe(true)
    }
  })

  it('declares explicit dimensions so React Flow need not measure first', () => {
    const { nodes } = buildLayout(payload, null)
    for (const node of nodes) {
      expect(node.width).toBe(NODE_WIDTH)
      expect(node.height).toBe(NODE_HEIGHT)
    }
  })

  it('gives every node a distinct position', () => {
    const { nodes } = buildLayout(payload, null)
    const seen = new Set(nodes.map((n) => `${n.position.x},${n.position.y}`))
    expect(seen.size).toBe(nodes.length)
  })
})

describe('buildLayout — supply-chain column order', () => {
  it('orders columns by the ontology, collapsing types absent from the graph', () => {
    const nodes = byId(buildLayout(payload, null).nodes)
    const columnOf = (id: string) => nodes[id].position.x / COLUMN_WIDTH

    // Columns follow COLUMN_ORDER, but a type with no nodes leaves no gap —
    // this fixture has no Incident, so Component moves up one column.
    const present = COLUMN_ORDER.filter((t) =>
      payload.nodes.some((n) => n.type === t),
    )
    expect(present).not.toContain('Incident')

    const expected: Record<string, string> = {
      'Yuki Tanaka': 'Employee',
      'Project Phoenix': 'Project',
      'C-17': 'Component',
      'Alpha Precision Systems': 'Supplier',
      'C-2048': 'Contract',
      'R-17': 'Risk',
    }
    for (const [id, type] of Object.entries(expected)) {
      expect(columnOf(id)).toBe(present.indexOf(type))
    }
  })

  it('leaves no empty column between populated ones', () => {
    const { nodes } = buildLayout(payload, null)
    const columns = [...new Set(nodes.map((n) => n.position.x / COLUMN_WIDTH))].sort(
      (a, b) => a - b,
    )
    expect(columns).toEqual(columns.map((_, i) => i))
  })

  it('lays the chain out strictly left to right', () => {
    const nodes = byId(buildLayout(payload, path).nodes)
    const xs = path.nodes.map((n) => nodes[n.id].position.x)
    for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeGreaterThan(xs[i - 1])
  })

  it('stacks same-type nodes on the row pitch', () => {
    const nodes = byId(buildLayout(payload, null).nodes)
    const gap = Math.abs(nodes['Project Titan'].position.y - nodes['Project Phoenix'].position.y)
    expect(gap).toBe(ROW_HEIGHT)
  })
})

describe('buildLayout — no path selected', () => {
  it('dims nothing and highlights nothing', () => {
    const { nodes, edges } = buildLayout(payload, null)
    for (const node of nodes) {
      const d = node.data as { highlighted: boolean; dimmed: boolean; pathIndex: number | null }
      expect(d.highlighted).toBe(false)
      expect(d.dimmed).toBe(false)
      expect(d.pathIndex).toBeNull()
    }
    for (const edge of edges) expect(edge.style?.opacity).toBe(1)
  })
})

describe('buildLayout — path highlighting', () => {
  it('highlights exactly the path nodes and dims the rest', () => {
    const { nodes } = buildLayout(payload, path)
    const onPath = new Set(path.nodes.map((n) => n.id))
    for (const node of nodes) {
      const d = node.data as { highlighted: boolean; dimmed: boolean }
      expect(d.highlighted).toBe(onPath.has(node.id))
      expect(d.dimmed).toBe(!onPath.has(node.id))
    }
  })

  it('numbers the path nodes in traversal order', () => {
    const nodes = byId(buildLayout(payload, path).nodes)
    path.nodes.forEach((n, index) => {
      expect((nodes[n.id].data as { pathIndex: number | null }).pathIndex).toBe(index)
    })
    expect((nodes['Project Titan'].data as { pathIndex: number | null }).pathIndex).toBeNull()
  })

  it('animates and thickens only the path edges', () => {
    const { edges } = buildLayout(payload, path)
    const onPath = new Set(['e1', 'e2', 'e3', 'e4'])
    for (const edge of edges) {
      if (onPath.has(edge.id)) {
        expect(edge.className).toBe('edge-animated')
        expect(edge.style?.stroke).toBe('#34d399')
        expect(edge.style?.opacity).toBe(1)
      } else {
        expect(edge.className).toBeUndefined()
        expect(edge.style?.opacity).toBeLessThan(1)
      }
    }
  })

  it('highlights a reverse-direction edge too', () => {
    const reversed: GraphPath = {
      ...path,
      steps: [
        { source: 'C-17', relation: 'uses', target: 'Project Phoenix', direction: 'reverse', docIds: ['d1'], evidence: '', phrase: '' },
      ],
      nodes: [path.nodes[1], path.nodes[0]],
      hops: 1,
      chain: 'C-17 -> Project Phoenix',
    }
    const edge = buildLayout(payload, reversed).edges.find((e) => e.id === 'e1')
    expect(edge?.className).toBe('edge-animated')
  })

  it('floats path members to the top of their column', () => {
    const nodes = byId(buildLayout(payload, path).nodes)
    // Phoenix is on the path, Titan is not — Phoenix must sit higher.
    expect(nodes['Project Phoenix'].position.y).toBeLessThan(nodes['Project Titan'].position.y)
  })
})

describe('buildLayout — edge cases', () => {
  it('handles an empty graph', () => {
    const { nodes, edges } = buildLayout({ nodes: [], edges: [], stats: {} }, null)
    expect(nodes).toEqual([])
    expect(edges).toEqual([])
  })

  it('puts an unknown entity type in the trailing column rather than dropping it', () => {
    const odd: GraphPayload = {
      nodes: [{ id: 'X', label: 'X', type: 'Martian', docIds: [], attributes: {} }],
      edges: [],
      stats: {},
    }
    const { nodes } = buildLayout(odd, null)
    expect(nodes).toHaveLength(1)
    expect(nodes[0].position.x).toBe(0)
  })

  it('ignores a path whose nodes are not in the graph', () => {
    const ghost: GraphPath = { nodes: [{ id: 'ghost', label: 'g', type: 'Risk' }], steps: [], hops: 0, chain: 'ghost' }
    const { nodes } = buildLayout(payload, ghost)
    expect(nodes.every((n) => !(n.data as { highlighted: boolean }).highlighted)).toBe(true)
  })
})
