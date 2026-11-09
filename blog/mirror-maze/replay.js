import {angleFor, forwardEdge, TOPOLOGIES} from './play/maze.js';

export const byId = id => document.getElementById(id);
export const number = value => value.toLocaleString(document.documentElement.lang);
export const fetchJson = async url => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  return response.json();
};

export function sceneFor(map) {
  return {...map, start:map.initial.cell, config:TOPOLOGIES[map.shape]};
}

export function decodeReplay(trace, map) {
  const scene = sceneFor(map);
  const exits = map.cells.map(cell => Array.from({length:map.headings}, (_, heading) => {
    const edge = forwardEdge(scene, cell.id, heading);
    return edge?.open ? edge.neighbor : -1;
  }));
  const total = trace.tape.length;
  const positions = new Uint16Array(total+1);
  const headings = new Uint8Array(total+1);
  const wallHits = new Uint32Array(total+1);
  const coverage = new Uint16Array(total+1);
  const cellVisits = new Set([map.initial.cell]);
  const seenEdges = new Set();
  const edges = [];
  let {cell, heading} = map.initial;
  positions[0] = cell;
  headings[0] = heading;
  coverage[0] = 1;

  for (let i=1; i<=total; i++) {
    const code = trace.tape[i-1];
    if (code==='L') heading = (heading+map.headings-1)%map.headings;
    else if (code==='R') heading = (heading+1)%map.headings;
    else if (code==='F') {
      const next = exits[cell][heading];
      if (next<0) throw new Error('Replay crosses a closed edge');
      const key = `${Math.min(cell,next)}:${Math.max(cell,next)}`;
      if (!seenEdges.has(key)) {
        seenEdges.add(key);
        edges.push([i,cell,next]);
      }
      cell = next;
      cellVisits.add(cell);
    } else if (code!=='W' || exits[cell][heading]!==-1) {
      throw new Error('Invalid recorded action');
    }
    positions[i] = cell;
    headings[i] = heading;
    wallHits[i] = wallHits[i-1] + Number(code==='W');
    coverage[i] = cellVisits.size;
  }

  return {
    scene, edges, wallHits, coverage, total,
    pose: index => ({cell:positions[index], heading:headings[index]}),
    code: index => trace.tape[index-1],
    roundFor(index) {
      if (!index) return 0;
      let lo = 1, hi = trace.rounds.length-1;
      while (lo<hi) {
        const mid = (lo+hi)>>1;
        if (trace.rounds[mid][2]<index) lo = mid+1;
        else hi = mid;
      }
      return lo;
    },
  };
}

export function drawMap(canvas, map, pose, edges=[], through=0, blocked=false) {
  const ctx = canvas.getContext('2d');
  const {width, height} = canvas;
  const vertices = map.cells.flatMap(cell => cell.vertices);
  const xs = vertices.map(point => point[0]), ys = vertices.map(point => point[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const scale = Math.min((width-60)/(xmax-xmin), (height-60)/(ymax-ymin));
  const project = ([x,y]) => [width/2+(x-(xmin+xmax)/2)*scale, height/2+(y-(ymin+ymax)/2)*scale];

  ctx.fillStyle = '#142b33';
  ctx.fillRect(0,0,width,height);
  ctx.lineJoin = ctx.lineCap = 'round';
  ctx.strokeStyle = '#92c3bb';
  ctx.lineWidth = 4;
  ctx.beginPath();
  for (const [index,a,b] of edges) {
    if (index>through) break;
    ctx.moveTo(...project(map.cells[a].center));
    ctx.lineTo(...project(map.cells[b].center));
  }
  ctx.stroke();
  ctx.strokeStyle = '#718b98';
  ctx.lineWidth = Math.max(1.5,width/400);
  ctx.beginPath();
  for (const cell of map.cells) {
    for (const edge of cell.edges) {
      if (edge.open || (edge.neighbor>=0 && edge.neighbor<cell.id)) continue;
      ctx.moveTo(...project(edge.a));
      ctx.lineTo(...project(edge.b));
    }
  }
  ctx.stroke();

  const scene = sceneFor(map);
  if (blocked) {
    const edge = forwardEdge(scene,pose.cell,pose.heading);
    if (edge) {
      ctx.strokeStyle = '#f08273';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(...project(edge.a));
      ctx.lineTo(...project(edge.b));
      ctx.stroke();
    }
  }
  for (const [id,color,radius] of [[map.initial.cell,'#d9af6e',6],[map.goal,'#a3d28f',9]]) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(...project(map.cells[id].center),radius,0,Math.PI*2);
    ctx.fill();
  }
  const [x,y] = project(map.cells[pose.cell].center);
  const angle = angleFor(scene,pose.heading);
  ctx.fillStyle = '#ffe9b3';
  ctx.strokeStyle = '#142b33';
  ctx.lineWidth = 2;
  ctx.beginPath();
  [0,2.5,-2.5].forEach((delta,i) => {
    const radius = i ? 10 : 17;
    const point = [x+radius*Math.cos(angle+delta),y+radius*Math.sin(angle+delta)];
    if (i) ctx.lineTo(...point);
    else ctx.moveTo(...point);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

export function metricCards(target, items) {
  target.replaceChildren(...items.map(([label,value]) => {
    const item = document.createElement('span'), count = document.createElement('b');
    count.textContent = value;
    item.append(label,count);
    return item;
  }));
}

export function timeline({mode, slider, previous, next, play, counter}, {
  labels, length, draw, positionFor,
}) {
  let timer = null;
  let activeMode = mode.value;
  const at = () => Number(slider.value);
  const max = () => Number(slider.max);
  function stop() {
    clearInterval(timer);
    timer = null;
    play.textContent = labels.play;
  }
  function show() {
    const index = at();
    previous.disabled = index<=0;
    next.disabled = index>=max();
    play.disabled = !max();
    counter.textContent = `${number(index)} / ${number(max())}`;
    draw(activeMode,index);
  }
  function seek(index) {
    slider.value = Math.min(max(),Math.max(0,index));
    show();
  }
  function reset(index=0) {
    stop();
    activeMode = mode.value;
    slider.max = length(activeMode);
    seek(index);
  }
  mode.onchange = () => {
    const index = positionFor(activeMode,at(),mode.value);
    reset(index);
  };
  slider.oninput = () => { stop(); show(); };
  previous.onclick = () => { stop(); seek(at()-1); };
  next.onclick = () => { stop(); seek(at()+1); };
  play.onclick = () => {
    if (timer) { stop(); return; }
    if (at()===max()) seek(0);
    play.textContent = labels.pause;
    timer = setInterval(() => {
      if (at()===max()) stop();
      else seek(at()+1);
    }, 420);
  };
  return {reset, stop, state:() => ({mode:activeMode,index:at()})};
}
