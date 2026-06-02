---
title: Baseline Methods Graph Interactive
---

# Baseline Methods Graph Interactive

This page provides a non-Mermaid interactive view of baseline-method relationships.

- Click a node to highlight its directly connected neighbors.
- Click the same node again to reset.
- Double-click a node, or `Ctrl/Cmd + click`, to open the corresponding paper card on Home.

<style>
  :root {
    --bg: #f4efe7;
    --surface: #fffaf3;
    --surface-strong: #ffffff;
    --border: #dfd3c2;
    --text: #2d241d;
    --muted: #6f6256;
    --accent: #c65d2e;
    --accent-soft: rgba(198, 93, 46, 0.12);
    --root: #2f6f5e;
    --root-soft: #dcefe8;
    --default: #4a6785;
    --default-soft: #e6edf5;
    --leaf: #9a5a4a;
    --leaf-soft: #f4e3dc;
    --edge: #b8aa97;
    --edge-active: #c65d2e;
    --dim: rgba(111, 98, 86, 0.18);
    --shadow: 0 14px 30px rgba(75, 52, 33, 0.08);
  }

  .graphx-shell {
    background:
      radial-gradient(circle at top left, rgba(204, 91, 44, 0.08), transparent 28%),
      radial-gradient(circle at top right, rgba(47, 95, 152, 0.08), transparent 26%),
      linear-gradient(180deg, #f8f2ea 0%, #f4efe7 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    box-shadow: var(--shadow);
    color: var(--text);
    margin: 24px 0 32px;
    overflow: hidden;
  }

  .graphx-header {
    align-items: flex-start;
    display: flex;
    gap: 20px;
    justify-content: space-between;
    padding: 28px 28px 18px;
  }

  .graphx-header h2 {
    font-size: 1.9rem;
    line-height: 1.1;
    margin: 0 0 8px;
  }

  .graphx-header p {
    color: var(--muted);
    margin: 0;
    max-width: 720px;
  }

  .graphx-grid {
    display: grid;
    gap: 18px;
    padding: 0 18px 20px;
  }

  .graphx-card {
    background: rgba(255, 250, 243, 0.88);
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: hidden;
  }

  .graphx-card.hidden {
    display: none;
  }

  .graphx-card-head {
    align-items: baseline;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: space-between;
    padding: 20px 22px 8px;
  }

  .graphx-head-main {
    min-width: 0;
  }

  .graphx-card-head h3 {
    font-size: 1.2rem;
    margin: 0;
    scroll-margin-top: 18px;
  }

  .graphx-meta {
    color: var(--muted);
    font-size: 0.92rem;
  }

  .graphx-zoom {
    align-items: center;
    display: inline-flex;
    gap: 8px;
  }

  .graphx-zoom-btn {
    appearance: none;
    background: var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text);
    cursor: pointer;
    font-size: 0.9rem;
    line-height: 1;
    min-width: 36px;
    padding: 8px 12px;
    transition: background 160ms ease, border-color 160ms ease;
  }

  .graphx-zoom-btn:hover {
    background: #fff;
    border-color: #c7b199;
  }

  .graphx-zoom-value {
    color: var(--muted);
    font-size: 0.86rem;
    font-variant-numeric: tabular-nums;
    min-width: 48px;
    text-align: center;
  }

  .graphx-canvas {
    overflow-x: auto;
    overscroll-behavior-x: contain;
    overscroll-behavior-y: auto;
    padding: 0 14px 18px;
    touch-action: pan-y pinch-zoom;
  }

  .graphx-svg {
    display: block;
  }

  .graphx-edge {
    fill: none;
    stroke: var(--edge);
    stroke-linecap: round;
    stroke-width: 2.2;
    transition: opacity 180ms ease, stroke 180ms ease, stroke-width 180ms ease;
  }

  .graphx-edge.dimmed {
    opacity: 0.14;
  }

  .graphx-edge.active {
    opacity: 1;
    stroke: var(--edge-active);
    stroke-width: 3.4;
  }

  .graphx-edge.secondary {
    opacity: 0.68;
    stroke: #d59a6a;
    stroke-width: 2.6;
  }

  .graphx-node {
    cursor: pointer;
    transition: opacity 180ms ease;
  }

  .graphx-node.dimmed {
    opacity: 0.22;
  }

  .graphx-node.active rect {
    stroke: var(--accent);
    stroke-width: 3;
  }

  .graphx-node.related rect {
    stroke: var(--accent);
    stroke-width: 2.2;
  }

  .graphx-node.related-2 rect {
    stroke: #d59a6a;
    stroke-width: 1.9;
  }

  .graphx-node rect {
    filter: drop-shadow(0 10px 18px rgba(45, 36, 29, 0.08));
    rx: 16;
    stroke-width: 1.5;
    transition: fill 180ms ease, stroke 180ms ease, stroke-width 180ms ease, filter 180ms ease;
  }

  .graphx-node:hover rect {
    filter: drop-shadow(0 12px 20px rgba(45, 36, 29, 0.12));
    stroke-width: 2.2;
  }

  .graphx-node.type-root rect {
    fill: var(--root-soft);
    stroke: var(--root);
  }

  .graphx-node.type-default rect {
    fill: var(--default-soft);
    stroke: var(--default);
  }

  .graphx-node.type-leaf rect {
    fill: var(--leaf-soft);
    stroke: var(--leaf);
  }

  .graphx-node text {
    fill: var(--text);
    pointer-events: none;
  }

  .graphx-node .graphx-title {
    font-size: 13px;
    font-weight: 700;
    text-anchor: middle;
  }

  .graphx-node .graphx-year {
    fill: var(--muted);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-anchor: middle;
  }

  .graphx-empty,
  .graphx-loading,
  .graphx-error {
    color: var(--muted);
    padding: 20px 28px 30px;
  }

  .graphx-error {
    color: #9b2c2c;
  }

  @media (max-width: 760px) {
    .graphx-header {
      flex-direction: column;
      padding: 24px 20px 16px;
    }

    .graphx-card-head {
      align-items: flex-start;
      flex-direction: column;
      padding-left: 16px;
      padding-right: 16px;
    }

    .graphx-canvas {
      padding: 0 8px 14px;
    }
  }
</style>

<div class="graphx-shell">
  <div class="graphx-header">
    <div>
      <h2>Interactive Family Graphs</h2>
      <p>Each family is rendered with native SVG and JavaScript. Clicking a node highlights only its directly connected neighborhood instead of the whole component.</p>
    </div>
  </div>

  <div id="graphx-loading" class="graphx-loading">Loading graph data...</div>
  <div id="graphx-error" class="graphx-error" hidden></div>
  <div id="graphx-grid" class="graphx-grid" hidden></div>
</div>

<script>
(() => {
  const dataUrl = new URL('../js/baseline_methods_graph_data.json', window.location.href).toString();
  const loadingEl = document.getElementById('graphx-loading');
  const errorEl = document.getElementById('graphx-error');
  const gridEl = document.getElementById('graphx-grid');

  const MIN_SCALE = 0.2;
  const MAX_SCALE = 1.8;
  const ZOOM_STEP = 1.15;
  const nodeSize = { width: 168, height: 54 };
  const state = {
    cards: [],
    activeNode: null,
    activeComponent: null,
  };
  const initialNodeId = new URLSearchParams(window.location.search).get('node');
  if (initialNodeId && window.location.hash) {
    window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);
  }
  if (initialNodeId && 'scrollRestoration' in window.history) {
    window.history.scrollRestoration = 'manual';
  }

  const setHashWithoutJump = (anchor) => {
    const url = new URL(window.location.href);
    url.hash = anchor ? `#${anchor}` : '';
    window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
  };

  const openPaperCard = (node) => {
    if (!node || !node.id) return;
    const url = new URL('../', window.location.href);
    url.searchParams.set('paper', node.id);
    window.open(url.toString(), '_blank', 'noopener');
  };

  const clampScale = (value) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, value));

  const clearActive = () => {
    state.cards.forEach((card) => {
      card.nodeGroups.forEach((group) => {
        group.classList.remove('active', 'related', 'related-2', 'dimmed');
      });
      card.edgePaths.forEach((path) => {
        path.classList.remove('active', 'secondary', 'dimmed');
      });
    });
    state.activeNode = null;
    state.activeComponent = null;
  };

  const setActiveNode = (card, node) => {
    const sameNode =
      state.activeNode &&
      state.activeComponent &&
      state.activeComponent.anchor === card.component.anchor &&
      state.activeNode.id === node.id;

    if (sameNode) {
      clearActive();
      return;
    }

    clearActive();

    const firstHopNodeIds = new Set(card.adjacency.get(node.id) || []);
    const secondHopNodeIds = new Set();

    firstHopNodeIds.forEach((firstHopId) => {
      const neighbors = card.adjacency.get(firstHopId) || [];
      neighbors.forEach((neighborId) => {
        if (neighborId !== node.id && !firstHopNodeIds.has(neighborId)) {
          secondHopNodeIds.add(neighborId);
        }
      });
    });

    card.nodeGroups.forEach((group, nodeId) => {
      if (nodeId === node.id) {
        group.classList.add('active');
      } else if (firstHopNodeIds.has(nodeId)) {
        group.classList.add('related');
      } else if (secondHopNodeIds.has(nodeId)) {
        group.classList.add('related-2');
      } else {
        group.classList.add('dimmed');
      }
    });

    card.edgePaths.forEach((path, edgeId) => {
      const edge = card.edgeMap.get(edgeId);
      if (!edge) return;
      const edgeNodes = [edge.source, edge.target];
      const touchesActive = edgeNodes.includes(node.id);
      const touchesFirstHop =
        edgeNodes.some((id) => firstHopNodeIds.has(id)) &&
        edgeNodes.some((id) => id === node.id || firstHopNodeIds.has(id));

      if (touchesActive) {
        path.classList.add('active');
      } else if (touchesFirstHop) {
        path.classList.add('secondary');
      } else {
        path.classList.add('dimmed');
      }
    });

    state.activeNode = node;
    state.activeComponent = card.component;
  };

  const focusNodeInView = (card, nodeId) => {
    const group = card.nodeGroups.get(nodeId);
    if (!group) return;

    requestAnimationFrame(() => {
      group.scrollIntoView({ block: 'center', inline: 'center', behavior: 'auto' });

      requestAnimationFrame(() => {
        if (card.canvas.scrollWidth <= card.canvas.clientWidth + 1) return;
        const canvasRect = card.canvas.getBoundingClientRect();
        const groupRect = group.getBoundingClientRect();
        const currentCenter = groupRect.left + groupRect.width / 2;
        const desiredCenter = canvasRect.left + canvasRect.width / 2;
        const deltaX = currentCenter - desiredCenter;
        if (Math.abs(deltaX) > 2) {
          card.canvas.scrollLeft += deltaX;
        }
      });
    });
  };

  const updateZoomDisplay = (card) => {
    if (card.zoomValue) {
      card.zoomValue.textContent = `${Math.round(card.scale * 100)}%`;
    }
  };

  const applyScale = (card, nextScale, anchorX = null) => {
    const previousScale = card.scale || 1;
    const scale = clampScale(nextScale);
    if (!card.svg) return;

    let contentAnchorX = null;
    if (anchorX !== null) {
      contentAnchorX = card.canvas.scrollLeft + anchorX;
    }

    card.scale = scale;
    card.svg.style.width = `${card.component.width * scale}px`;
    card.svg.style.height = `${card.component.height * scale}px`;
    updateZoomDisplay(card);

    if (contentAnchorX !== null) {
      const ratio = scale / previousScale;
      const nextScrollLeft = contentAnchorX * ratio - anchorX;
      card.canvas.scrollLeft = Math.max(0, nextScrollLeft);
    }
  };

  const getFitScale = (card) => {
    const availableWidth = card.canvas.clientWidth;
    if (!availableWidth || !card.component.width) return 1;
    return clampScale(Math.min(1, availableWidth / card.component.width));
  };

  const fitCardToCanvas = (card) => {
    card.userScaled = false;
    applyScale(card, getFitScale(card), 0);
  };

  const wrapTitleLines = (label) => {
    const plain = String(label || '').replace(/\[\d{4}\]$/, '').trim();
    const words = plain.split(/\s+/).filter(Boolean);
    if (words.length <= 1 && plain.length <= 16) {
      return [plain];
    }
    const lines = [];
    let current = '';
    words.forEach((word) => {
      const candidate = current ? `${current} ${word}` : word;
      if (candidate.length <= 16 || !current) {
        current = candidate;
      } else {
        lines.push(current);
        current = word;
      }
    });
    if (current) {
      lines.push(current);
    }
    return lines.slice(0, 2);
  };

  const makeSvgEl = (tag, attrs = {}) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    return el;
  };

  const buildEdgePath = (source, target) => {
    const startX = source.x + nodeSize.width / 2;
    const startY = source.y;
    const endX = target.x - nodeSize.width / 2;
    const endY = target.y;
    const dx = Math.max(36, (endX - startX) * 0.45);
    return `M ${startX} ${startY} C ${startX + dx} ${startY}, ${endX - dx} ${endY}, ${endX} ${endY}`;
  };

  const buildCard = (component) => {
    const card = document.createElement('section');
    card.className = 'graphx-card';
    card.id = component.anchor;

    const head = document.createElement('div');
    head.className = 'graphx-card-head';

    const headMain = document.createElement('div');
    headMain.className = 'graphx-head-main';

    const title = document.createElement('h3');
    title.textContent = `${component.title} Family`;

    const meta = document.createElement('div');
    meta.className = 'graphx-meta';
    meta.textContent = `${component.node_count} methods · ${component.edge_count} relationships`;

    headMain.append(title, meta);

    const zoom = document.createElement('div');
    zoom.className = 'graphx-zoom';

    const zoomOutBtn = document.createElement('button');
    zoomOutBtn.className = 'graphx-zoom-btn';
    zoomOutBtn.type = 'button';
    zoomOutBtn.textContent = '-';
    zoomOutBtn.title = 'Zoom out';

    const zoomValue = document.createElement('div');
    zoomValue.className = 'graphx-zoom-value';
    zoomValue.textContent = '100%';

    const fitBtn = document.createElement('button');
    fitBtn.className = 'graphx-zoom-btn';
    fitBtn.type = 'button';
    fitBtn.textContent = 'Fit';
    fitBtn.title = 'Fit to width';

    const zoomInBtn = document.createElement('button');
    zoomInBtn.className = 'graphx-zoom-btn';
    zoomInBtn.type = 'button';
    zoomInBtn.textContent = '+';
    zoomInBtn.title = 'Zoom in';

    zoom.append(zoomOutBtn, zoomValue, fitBtn, zoomInBtn);
    head.append(headMain, zoom);

    const canvas = document.createElement('div');
    canvas.className = 'graphx-canvas';

    const svg = makeSvgEl('svg', {
      class: 'graphx-svg',
      viewBox: `0 0 ${component.width} ${component.height}`,
      width: String(component.width),
      height: String(component.height),
      role: 'img',
      'aria-label': `${component.title} family graph`,
      preserveAspectRatio: 'xMinYMin meet',
    });

    const defs = makeSvgEl('defs');
    const marker = makeSvgEl('marker', {
      id: `arrow-${component.anchor}`,
      markerWidth: '10',
      markerHeight: '10',
      refX: '9',
      refY: '5',
      orient: 'auto-start-reverse',
      markerUnits: 'strokeWidth',
    });
    marker.appendChild(makeSvgEl('path', {
      d: 'M 0 0 L 10 5 L 0 10 z',
      fill: 'context-stroke',
    }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    const edgeLayer = makeSvgEl('g');
    const nodeLayer = makeSvgEl('g');
    svg.append(edgeLayer, nodeLayer);
    canvas.appendChild(svg);

    card.append(head, canvas);

    const nodeMap = new Map(component.nodes.map((node) => [node.id, node]));
    const nodeGroups = new Map();
    const edgePaths = new Map();
    const edgeMap = new Map();
    const adjacency = new Map();

    component.nodes.forEach((node) => {
      adjacency.set(node.id, new Set());
    });

    component.edges.forEach((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) return;

      adjacency.get(edge.source).add(edge.target);
      adjacency.get(edge.target).add(edge.source);

      const edgeId = `${edge.source}__${edge.target}`;
      edgeMap.set(edgeId, edge);
      const path = makeSvgEl('path', {
        class: 'graphx-edge',
        d: buildEdgePath(source, target),
        'marker-end': `url(#arrow-${component.anchor})`,
      });
      edgeLayer.appendChild(path);
      edgePaths.set(edgeId, path);
    });

    component.nodes.forEach((node) => {
      const group = makeSvgEl('g', {
        class: `graphx-node type-${node.type}`,
        transform: `translate(${node.x - nodeSize.width / 2} ${node.y - nodeSize.height / 2})`,
        tabindex: '0',
        role: 'button',
        'aria-label': `Highlight neighbors of ${node.label}`,
      });

      const rect = makeSvgEl('rect', {
        width: String(nodeSize.width),
        height: String(nodeSize.height),
      });
      group.appendChild(rect);

      const titleLines = wrapTitleLines(node.display_name || node.label);
      const textTop = titleLines.length === 1 ? 23 : 18;
      titleLines.forEach((line, index) => {
        const t = makeSvgEl('text', {
          class: 'graphx-title',
          x: String(nodeSize.width / 2),
          y: String(textTop + index * 15),
        });
        t.textContent = line;
        group.appendChild(t);
      });

      if (node.year_label) {
        const year = makeSvgEl('text', {
          class: 'graphx-year',
          x: String(nodeSize.width / 2),
          y: titleLines.length === 1 ? '39' : '46',
        });
        year.textContent = node.year_label;
        group.appendChild(year);
      }

      const onSelect = () => setActiveNode(cardRef, node);
      group.addEventListener('click', onSelect);
      group.addEventListener('dblclick', (event) => {
        event.preventDefault();
        event.stopPropagation();
        openPaperCard(node);
      });
      group.addEventListener('click', (event) => {
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          event.stopPropagation();
          openPaperCard(node);
        }
      });
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onSelect();
        }
        if ((event.key === 'o' || event.key === 'O') && (event.metaKey || event.ctrlKey)) {
          event.preventDefault();
          openPaperCard(node);
        }
      });

      nodeLayer.appendChild(group);
      nodeGroups.set(node.id, group);
    });

    const cardRef = {
      component,
      card,
      canvas,
      svg,
      zoomValue,
      zoomOutBtn,
      fitBtn,
      zoomInBtn,
      nodeGroups,
      edgePaths,
      edgeMap,
      adjacency,
      scale: 1,
      userScaled: false,
    };

    const zoomAroundCenter = (factor) => {
      cardRef.userScaled = true;
      applyScale(cardRef, cardRef.scale * factor, cardRef.canvas.clientWidth / 2);
    };

    zoomOutBtn.addEventListener('click', () => zoomAroundCenter(1 / ZOOM_STEP));
    zoomInBtn.addEventListener('click', () => zoomAroundCenter(ZOOM_STEP));
    fitBtn.addEventListener('click', () => fitCardToCanvas(cardRef));

    canvas.addEventListener('wheel', (event) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      cardRef.userScaled = true;
      const rect = canvas.getBoundingClientRect();
      const anchorX = event.clientX - rect.left;
      const factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
      applyScale(cardRef, cardRef.scale * factor, anchorX);
    }, { passive: false });

    canvas.addEventListener('gesturestart', (event) => {
      event.preventDefault();
      cardRef.gestureStartScale = cardRef.scale;
    }, { passive: false });

    canvas.addEventListener('gesturechange', (event) => {
      event.preventDefault();
      cardRef.userScaled = true;
      const rect = canvas.getBoundingClientRect();
      const anchorX = event.clientX ? event.clientX - rect.left : rect.width / 2;
      applyScale(cardRef, (cardRef.gestureStartScale || cardRef.scale) * event.scale, anchorX);
    }, { passive: false });

    return cardRef;
  };

  const revealAndSelectNode = (nodeId) => {
    if (!nodeId) return;
    const card = state.cards.find((item) => item.nodeGroups.has(nodeId));
    if (!card) return false;
    const node = card.component.nodes.find((item) => item.id === nodeId);
    if (!node) return false;

    card.card.classList.remove('hidden');
    setHashWithoutJump(card.component.anchor);
    setActiveNode(card, node);
    focusNodeInView(card, node.id);
    return true;
  };

  const settleInitialNodeFocus = (nodeId) => {
    const attempts = [0, 120, 360];
    attempts.forEach((delay, index) => {
      setTimeout(() => {
        const found = revealAndSelectNode(nodeId);
        if (index === attempts.length - 1 || !found) {
          gridEl.style.opacity = '';
          gridEl.style.pointerEvents = '';
          loadingEl.hidden = true;
          if (!found) {
            errorEl.hidden = false;
            errorEl.textContent = `Node not found: ${nodeId}`;
          }
        }
      }, delay);
    });
  };

  const render = (payload) => {
    const components = Array.isArray(payload.components) ? payload.components : [];
    gridEl.innerHTML = '';
    state.cards = components.map((component) => buildCard(component));
    state.cards.forEach((card) => {
      gridEl.appendChild(card.card);
    });
    gridEl.hidden = false;
    state.cards.forEach((card) => {
      fitCardToCanvas(card);
    });

    if (!initialNodeId && window.location.hash) {
      const target = document.getElementById(window.location.hash.slice(1));
      if (target) {
        target.scrollIntoView({ block: 'start' });
      }
    }

    if (initialNodeId) {
      loadingEl.textContent = 'Opening selected node...';
      gridEl.style.opacity = '0';
      gridEl.style.pointerEvents = 'none';

      const runInitialFocus = () => {
        settleInitialNodeFocus(initialNodeId);
      };

      if (document.readyState === 'complete') {
        runInitialFocus();
      } else {
        window.addEventListener('load', runInitialFocus, { once: true });
      }
    } else {
      loadingEl.hidden = true;
    }
  };

  fetch(dataUrl)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to load graph data: ${response.status}`);
      }
      return response.json();
    })
    .then(render)
    .catch((error) => {
      loadingEl.hidden = true;
      errorEl.hidden = false;
      errorEl.textContent = error.message || 'Failed to load graph data.';
    });

  window.addEventListener('resize', () => {
    state.cards.forEach((card) => {
      if (!card.userScaled) {
        fitCardToCanvas(card);
      }
    });
  });
})();
</script>
