import {angleFor} from './play/maze.js';
import {MirrorRenderer} from './play/renderer.js';
import {byId as $, number, fetchJson, decodeReplay, drawMap, metricCards, timeline} from './replay.js';

const zh = document.documentElement.lang === 'zh-CN';
const t = (en,cn) => zh ? cn : en;
const root = document.body.dataset.root;
const read = path => fetchJson(root+path);
const labels = {
  success:t('Goal reached','已通关'), planning_cap:t('Planning cap','达到规划上限'),
  quota_interrupted:t('Not evaluated','未完成评测'), pending:t('API interrupted','API 暂停'),
  skipped:t('Policy skip','按规则跳过'), withdrawn:t('Withdrawn','已撤回'),
  easy:t('Easy','简单'), medium:t('Medium','中等'), hard:t('Hard','困难'),
  square:t('Square','正方形'), hex:t('Hexagonal','六边形'),
};
const commands = {
  L:t('Turn left','左转'), R:t('Turn right','右转'),
  F:t('Forward · moved','前进 · 移动成功'), W:t('Forward · wall hit','前进 · 撞墙'),
};
const query = new URLSearchParams(location.search);
const filterKeys = ['difficulty','mode','model','shape'];
const execution = {harness:'Harness',api:t('Direct API','直接 API')};
let replayRows = [], visibleRows = [], selectedId;
const filter = key => $(`filter-${key}`);
const filterValues = () => Object.fromEntries(filterKeys.map(key => [key,filter(key).value]));
const defaultTrial = 'harness-hard-gpt-hex-MIRROR-043';
let study, hero, heroIndex=0, revealed=false, current, trace, map, decoded;
let controller, renderer, rendererFailed=false, lastPose, generation=0;
const maps = new Map();
function readMap(key) {
  if (!maps.has(key)) maps.set(key,read(`data/maps/${key}.json`));
  return maps.get(key);
}
function setText(id,text) { $(id).textContent = text; }

async function showHero() {
  const view = hero.views[heroIndex];
  $('hero-image').src = root+view.image;
  $('view-a').setAttribute('aria-pressed',String(heroIndex===0));
  $('view-b').setAttribute('aria-pressed',String(heroIndex===1));
  $('hero-map').hidden = !revealed;
  setText('reveal',revealed ? t('Hide the geometry','隐藏结构') : t('Reveal the geometry','揭晓结构'));
  if (revealed) drawMap($('hero-map'),await readMap(hero.map),view);
  setText('hero-answer',!revealed ? t('Two recorded headings in the same cell. Choose a view and reveal the answer.','两张图来自同一格的不同朝向。选择一个视角，再揭晓答案。') :
    view.open ? t('There is a real opening directly ahead.','正前方有真实开口。') :
    t('A mirror wall blocks the edge. The apparent corridor is a reflection.','正前方是镜面墙。看似延伸的通道来自反射。'));
}

function resultTable() {
  for (const [id,name] of Object.entries(study.models)) {
    const rows = study.rows.filter(row => row.model===id && row.mode==='harness');
    if (!rows.length) continue;
    const tr = $('native-results').insertRow();
    tr.insertCell().textContent = name;
    tr.insertCell().textContent = rows[0].client;
    for (const level of ['easy','medium','hard']) {
      const runs = rows.filter(row => row.difficulty===level);
      const done = runs.filter(row => row.auditedOutcome);
      const cell = tr.insertCell();
      const score = document.createElement('span');
      score.className = 'score' + (done.length ? '' : ' unavailable');
      score.textContent = done.length ? `${done.filter(row=>row.status==='success').length}/${done.length}` :
        runs.every(row=>row.status==='skipped') ? labels.skipped : labels.quota_interrupted;
      const dots = document.createElement('div');
      dots.className = 'dots';
      for (const row of runs) {
        const dot = document.createElement(row.trace ? 'button' : 'span');
        dot.className = 'dot '+row.status;
        dot.title = `${labels[row.shape]} ${row.seed} · ${labels[row.status]}`;
        if (row.trace) {
          dot.setAttribute('aria-label',`${name} · ${dot.title}`);
          dot.onclick = () => jumpToTrial(row.id);
        }
        dots.append(dot);
      }
      cell.append(score,dots);
    }
  }
}

function stepFor(mode,index) {
  if (mode==='action') return {action:index, round:decoded.roundFor(index)};
  const round = mode==='sample' ? trace.frames[index].round : index;
  return {action:trace.rounds[round][2],round};
}
function convertMode(from,index,to) {
  const {action,round} = stepFor(from,index);
  if (to==='action') return action;
  if (to==='round') return round;
  return Math.max(0,trace.frames.findIndex((frame,i) =>
    frame.round<=round && (i===trace.frames.length-1 || trace.frames[i+1].round>round)));
}
function showObservation(mode,index,round,action,pose) {
  const exact = mode==='sample' ? trace.frames[index] :
    trace.frames.find(frame => frame.round===round && trace.rounds[round][2]===action);
  if (!exact && !rendererFailed) {
    try {
      if (!renderer) {
        renderer = new MirrorRenderer($('replay-render'));
        renderer.setResolution(640,360);
        renderer.setMaze(decoded.scene);
      }
      const key = `${current.map}:${pose.cell}:${pose.heading}`;
      if (lastPose!==key) {
        renderer.render({
          cell:pose.cell, position:map.cells[pose.cell].center,
          angle:angleFor(decoded.scene,pose.heading), mirror:true, bounces:10,
        });
        lastPose = key;
      }
      $('replay-image').hidden = true;
      $('replay-render').hidden = false;
      setText('image-round',t('Reconstructed from the recorded pose','按记录位置和朝向重渲染'));
      return;
    } catch {
      rendererFailed = true;
    }
  }
  const sample = exact ?? trace.frames.filter(frame => frame.round<=round).at(-1) ?? trace.frames[0];
  $('replay-render').hidden = true;
  $('replay-image').hidden = false;
  $('replay-image').src = root+sample.image;
  setText('image-round',exact ?
    t(`Recorded view · after round ${sample.round}`,`原始画面 · 第 ${sample.round} 轮结束`) :
    t(`Nearest recorded view · round ${sample.round}`,`最近的原始画面 · 第 ${sample.round} 轮`));
}
function updateCoverage(round,visited) {
  const x = 35 + round/Math.max(1,trace.rounds.length-1)*850;
  const y = 145 - visited/current.cells*130;
  $('coverage-marker').setAttribute('x1',x);
  $('coverage-marker').setAttribute('x2',x);
  $('coverage-point').setAttribute('cx',x);
  $('coverage-point').setAttribute('cy',y);
}
function draw(mode,index) {
  if (!decoded) return;
  const {action,round} = stepFor(mode,index);
  const pose = decoded.pose(action);
  const hit = mode==='action' && decoded.code(action)==='W';
  showObservation(mode,index,round,action,pose);
  drawMap($('replay-map'),map,pose,decoded.edges,action,hit);
  const counter = round===0 ? t('Initial pose','起始位置') :
    mode==='action' ? commands[decoded.code(action)] : t('End of planning round','本轮规划结束');
  setText('step-action',counter);
  $('step-action').classList.toggle('wall-hit',hit);
  metricCards($('replay-metrics'),[
    [t('Planning round','规划轮次'),`${number(round)} / ${number(current.cap)}`],
    [t('Primitive actions','基础动作'),number(action)],
    [t('Physical wall hits','实际撞墙'),number(decoded.wallHits[action])],
    [t('Cells visited','已探索格子'),`${decoded.coverage[action]} / ${current.cells}`],
  ]);
  updateCoverage(round,decoded.coverage[action]);
}
function markSelected(scrollList=false) {
  for (const button of $('trial-list').querySelectorAll('button')) {
    const selected = button.dataset.record===selectedId;
    if (selected) {
      button.setAttribute('aria-current','true');
      if (scrollList) {
        const list = $('trial-list');
        list.scrollTop = button.offsetTop - (list.clientHeight-button.offsetHeight)/2;
      }
    } else button.removeAttribute('aria-current');
  }
}
function renderTrialList() {
  const values = filterValues();
  visibleRows = replayRows.filter(row => filterKeys.every(key => !values[key] || row[key]===values[key]));
  setText('trial-count',t(
    `${visibleRows.length} of ${replayRows.length} records`,
    `显示 ${visibleRows.length} / ${replayRows.length} 条记录`
  ));
  $('clear-filters').disabled = !Object.values(values).some(Boolean);
  $('no-trials').hidden = visibleRows.length>0;
  $('trial-list').hidden = !visibleRows.length;
  $('trial-list').replaceChildren(...visibleRows.map(row => {
    const item = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.record = row.id;
    const name = document.createElement('strong');
    name.textContent = row.name;
    const subtitle = document.createElement('span');
    const mode = row.mode==='api' ? execution.api : `${execution.harness} · ${row.client}`;
    subtitle.textContent = `${mode} · ${labels[row.difficulty]} · ${labels[row.shape]} · ${row.seed}`;
    const badge = document.createElement('span');
    badge.className = 'badge '+row.status;
    badge.textContent = labels[row.status];
    button.append(name,subtitle,badge);
    button.onclick = () => {
      if (row.id!==selectedId) void openTrial(row.id);
    };
    item.append(button);
    return item;
  }));
  markSelected(true);
}
function syncFilters(requested,step=0) {
  renderTrialList();
  if (!visibleRows.length) {
    ++generation;
    controller.stop();
    current = null;
    decoded = null;
    selectedId = null;
    $('replay-body').hidden = true;
    $('replay-loading').hidden = true;
    return;
  }
  const next = [requested,selectedId].find(id => visibleRows.some(row => row.id===id)) ?? visibleRows[0].id;
  if (next!==selectedId) return openTrial(next,false,step);
}
function setupFilters() {
  const options = {
    difficulty:['easy','medium','hard'].map(key => [key,labels[key]]),
    mode:Object.entries(execution),
    model:[...new Map(replayRows.map(row => [row.model,row.name]))].sort((a,b)=>a[1].localeCompare(b[1])),
    shape:['square','hex'].map(key => [key,labels[key]]),
  };
  for (const key of filterKeys) {
    filter(key).append(new Option(t('All','全部'),''));
    for (const [value,label] of options[key]) filter(key).append(new Option(label,value));
    const value = query.get(key);
    if (options[key].some(option => option[0]===value)) filter(key).value = value;
    filter(key).onchange = () => syncFilters();
  }
  $('clear-filters').onclick = () => {
    for (const key of filterKeys) filter(key).value = '';
    syncFilters();
  };
}
function jumpToTrial(id) {
  if (!visibleRows.some(row => row.id===id)) {
    for (const key of filterKeys) filter(key).value = '';
    renderTrialList();
  }
  return openTrial(id,true);
}

async function openTrial(id,scroll=false,index=0) {
  const row = visibleRows.find(item => item.id===id);
  if (!row) return;
  const ticket = ++generation;
  controller.stop();
  $('replay-body').hidden = true;
  $('replay-loading').hidden = false;
  setText('replay-loading',t('Loading trajectory…','正在加载轨迹…'));
  selectedId = id;
  markSelected(true);
  if (scroll) $('replay').scrollIntoView({behavior:'smooth'});
  try {
    const [packet,geometry] = await Promise.all([read(row.trace),readMap(row.map)]);
    if (ticket!==generation) return;
    const state = decodeReplay(packet,geometry);
    current = row;
    trace = packet;
    map = geometry;
    decoded = state;
    lastPose = undefined;
    if (renderer && !rendererFailed) renderer.setMaze(decoded.scene);
    $('replay-loading').hidden = true;
    $('replay-body').hidden = false;
    setText('active-trial',`${row.name} · ${row.client} · ${labels[row.shape]} ${row.seed}`);
    $('trial-status').className = 'badge '+row.status;
    setText('trial-status',labels[row.status]);
    setText('replay-caption',row.originalCap!==row.cap ?
      t(`Original declared budget was ${row.originalCap} calls. Review frames are sampled; action and round positions are exact.`,
        `原始声明预算为 ${row.originalCap} 轮。画面为采样记录，动作及轮次位置可精确回放。`) :
      t('Step through exact recorded actions. Image samples preserve the original observations.',
        '可逐个查看精确记录的动作。采样画面保留原始观测。'));
    const x = i => 35+i/Math.max(1,trace.rounds.length-1)*850;
    const y = v => 145-v/current.cells*130;
    $('coverage-line').setAttribute('d',trace.rounds.map((r,i)=>`${i?'L':'M'}${x(i)},${y(r[4])}`).join(' '));
    controller.reset(index);
  } catch (error) {
    if (ticket!==generation) return;
    $('replay-loading').hidden = false;
    setText('replay-loading',t('This trajectory could not be loaded.','该轨迹暂时无法加载。'));
    selectedId = null;
    current = null;
    decoded = null;
    markSelected();
    console.error(error);
  }
}
function citation() {
  $('copy-citation').onclick = async () => {
    const citation = $('bibtex').textContent;
    try {
      if (navigator.clipboard) await navigator.clipboard.writeText(citation);
      else {
        const field = document.createElement('textarea');
        field.value = citation;
        field.className = 'copy-buffer';
        document.body.append(field);
        field.select();
        const copied = document.execCommand('copy');
        field.remove();
        if (!copied) throw new Error('Clipboard unavailable');
      }
      setText('copy-status',t('Copied','已复制'));
    } catch {
      const range = document.createRange();
      range.selectNodeContents($('bibtex'));
      window.getSelection().removeAllRanges();
      window.getSelection().addRange(range);
      setText('copy-status',t('Selected. Use your copy shortcut.','已选中文本，请用快捷键复制。'));
    }
  };
}

async function init() {
  [study,hero] = await Promise.all([read('data/study.json'),read('data/hero.json')]);
  resultTable();
  const difficultyOrder = ['easy','medium','hard'];
  replayRows = study.rows.filter(row=>row.trace).sort((a,b) =>
    difficultyOrder.indexOf(a.difficulty)-difficultyOrder.indexOf(b.difficulty) ||
    a.name.localeCompare(b.name) || a.mode.localeCompare(b.mode) ||
    a.shape.localeCompare(b.shape) || a.seed.localeCompare(b.seed)
  );
  const ui = {
    mode:$('replay-mode'), slider:$('replay-slider'), previous:$('previous'),
    next:$('next'), play:$('play-toggle'), counter:$('step-counter'),
  };
  const unit = query.get('stepMode') ?? query.get('mode');
  if (['action','round','sample'].includes(unit)) ui.mode.value = unit;
  controller = timeline(ui,{
    labels:{play:t('Play','播放'),pause:t('Pause','暂停')},
    length:mode => mode==='action' ? decoded.total : mode==='round' ? trace.rounds.length-1 : trace.frames.length-1,
    draw, positionFor:convertMode,
  });
  setupFilters();
  for (const id of ['view-a','view-b']) $(id).onclick = () => { heroIndex = Number(id==='view-b'); showHero(); };
  $('reveal').onclick = () => { revealed = !revealed; showHero(); };
  document.querySelectorAll('[data-trial]').forEach(button => {
    button.onclick = () => jumpToTrial(button.dataset.trial);
  });
  $('language-switch').onclick = () => {
    const next = new URL($('language-switch').href);
    const active = Object.entries(filterValues()).filter(([,value])=>value);
    if (selectedId) {
      const {mode,index} = controller.state();
      active.push(['trial',selectedId],['stepMode',mode],['step',index]);
    }
    next.search = new URLSearchParams(active);
    next.hash = location.hash;
    $('language-switch').href = next.href;
  };
  citation();
  const requested = query.get('trial') ?? defaultTrial;
  const index = Number(query.get('step'));
  await Promise.all([showHero(),syncFilters(requested,Number.isFinite(index) ? Math.floor(index) : 0)]);
  document.body.dataset.ready = 'true';
}
init().catch(error => {
  console.error(error);
  document.body.dataset.error = error.message;
});
