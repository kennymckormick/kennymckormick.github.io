import {byId as $, number, fetchJson, decodeReplay, drawMap, metricCards, timeline} from '../replay.js';

const zh = document.documentElement.lang === 'zh-CN';
const t = (en,cn) => zh ? cn : en;
const root = document.body.dataset.root;
const read = path => fetchJson(root+path);
const maps = new Map();
const levels = {easy:t('Easy','简单'), medium:t('Medium','中等')};
const shapes = {square:t('Square','正方形'), hex:t('Hexagonal','六边形')};
const actions = {
  L:t('Turn left','左转'), R:t('Turn right','右转'),
  F:t('Forward · moved','前进 · 移动成功'), W:t('Forward · wall hit','前进 · 撞墙'),
};
let study, current, map, decoded, controller, generation=0;
const params = new URLSearchParams(location.search);

function draw(mode,index) {
  if (!current || !map) return;
  const step = mode==='action' ? index : current.rounds[index][2];
  const round = mode==='round' ? index : decoded.roundFor(step);
  const hit = mode==='action' && decoded.code(step)==='W';
  drawMap($('text-map'),map,decoded.pose(step),decoded.edges,step,hit);
  metricCards($('text-metrics'),[
    [t('Planning rounds','规划轮次'),`${round} / ${current.cap}`],
    [t('Primitive actions','基础动作'),number(step)],
    [t('Wall hits','实际撞墙'),number(decoded.wallHits[step])],
    [t('Cells visited','已探索格子'),`${decoded.coverage[step]} / ${current.cells}`],
  ]);
  $('text-action').textContent = round===0 ? t('Initial pose','起始位置') :
    mode==='action' ? actions[decoded.code(step)] : t('End of planning round','本轮规划结束');
  $('text-action').classList.toggle('wall-hit',hit);
}

async function openTrial(id,index=0) {
  controller.stop();
  const ticket = ++generation;
  const trial = study.rows.find(row=>row.id===id);
  if (!maps.has(trial.map_key)) maps.set(trial.map_key,read(`data/maps/${trial.map_key}.json`));
  const maze = await maps.get(trial.map_key);
  if (ticket!==generation) return;
  current = trial;
  map = maze;
  decoded = decodeReplay(current,map);
  $('text-status').className = `badge ${trial.status}`;
  $('text-status').textContent = trial.status==='success' ? t('Goal reached','已通关') : t('Planning cap','达到规划上限');
  controller.reset(index);
}

function showBudget(pairs) {
  const maxWalls = Number($('walls').value);
  const maxActions = Number($('actions').value);
  const counts = ['visual','text'].map(mode => pairs.filter(pair => {
    const row = pair[mode];
    return row.status==='success' && row.walls<=maxWalls && row.actions<=maxActions;
  }).length);
  $('budget-score').textContent = `${t('Visual','视觉')} ${counts[0]}/${pairs.length} · ${t('Text','纯文本')} ${counts[1]}/${pairs.length}`;
}

function showVideos(manifest) {
  $('case-videos').replaceChildren(...manifest.videos.map(item => {
    const figure = document.createElement('figure');
    const video = document.createElement('video');
    const caption = document.createElement('figcaption');
    const title = zh ? item.title.replace('Easy','简单').replace('Medium','中等').replace('Square','正方形').replace('Hexagonal','六边形') : item.title;
    video.src = root+item.file;
    video.controls = true;
    video.preload = 'metadata';
    video.setAttribute('playsinline','');
    video.setAttribute('aria-label',title);
    caption.textContent = title;
    figure.append(video,caption);
    return figure;
  }));
}

async function init() {
  const [text,budget,videos] = await Promise.all([
    read('data/text-study.json'),read('data/text-visual-budget.json'),read('data/text-videos.json'),
  ]);
  study = text;
  $('text-trial').replaceChildren(...study.rows.map(row => new Option(
    `${study.models[row.model]} · ${levels[row.difficulty]} ${shapes[row.shape]} ${row.seed}`,row.id
  )));
  const ui = {
    mode:$('text-mode'), slider:$('text-round'), previous:$('text-previous'),
    next:$('text-next'), play:$('text-play'), counter:$('text-step-counter'),
  };
  if (params.get('mode')==='round') ui.mode.value = 'round';
  controller = timeline(ui,{
    labels:{play:t('Play','播放'),pause:t('Pause','暂停')},
    length:mode=>mode==='action' ? decoded.total : current.rounds.length-1,
    positionFor:(from,index)=>from==='action' ? decoded.roundFor(index) : current.rounds[index][2],
    draw,
  });
  $('text-trial').onchange = event => openTrial(event.target.value);
  $('walls').oninput = $('actions').oninput = () => showBudget(budget.pairs);
  $('language-switch').onclick = () => {
    const next = new URL($('language-switch').href);
    if (current && location.hash==='#replay') {
      const {mode,index} = controller.state();
      next.search = new URLSearchParams({trial:current.id,mode,step:index});
    }
    next.hash = location.hash;
    $('language-switch').href = next.href;
  };
  showBudget(budget.pairs);
  showVideos(videos);
  const id = study.rows.find(row=>row.id===params.get('trial'))?.id ?? study.rows[0].id;
  $('text-trial').value = id;
  const index = Number(params.get('step'));
  await openTrial(id,Number.isFinite(index) ? Math.floor(index) : 0);
  document.body.dataset.ready = 'true';
}

init().catch(error => {
  console.error(error);
  document.body.dataset.error = error.message;
});
