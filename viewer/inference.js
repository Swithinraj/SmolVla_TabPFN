const $ = id => document.getElementById(id);
const cameras = ['observation.images.image', 'observation.images.image2'];
const videos = [$('agent'), $('wrist')];
let episodes = [], totalEpisodes = 0, current = null, predictions = null;
let playing = false, frame = 0, animation = 0, requestToken = 0;

function timeText(t) {
  const seconds = Math.floor(Math.max(t, 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}
function numberText(value) { return Number.isFinite(value) ? value.toFixed(4) : '—'; }
function stop() { playing = false; videos.forEach(video => video.pause()); $('play').textContent = '▶'; cancelAnimationFrame(animation); }
function timeAt(camera) { return current.videos[cameras[camera]].start + frame / current.fps; }
function seekVideos() { videos.forEach((video, camera) => { if (video.readyState && Math.abs(video.currentTime - timeAt(camera)) > 0.055) video.currentTime = timeAt(camera); }); }
function predictionAtFrame() { return predictions?.frames?.[frame] || null; }

function showFrame(next, seek = true) {
  if (!current) return;
  frame = Math.max(0, Math.min(current.frames - 1, Math.round(next)));
  $('seek').value = frame;
  $('clock').textContent = `${timeText(frame / current.fps)} / ${timeText(current.duration)}`;
  $('frameLabel').textContent = `Frame ${frame} / ${current.frames - 1}`;
  $('stateLabel').textContent = `Frame ${frame}`;
  const prediction = predictionAtFrame(), groundTruth = current.action[frame];
  $('frameLoss').textContent = numberText(prediction?.loss);
  $('frameMae').textContent = numberText(prediction?.mae);
  $('actions').innerHTML = groundTruth.map((value, dimension) => `<tr><td>A${dimension + 1}</td><td class="gt">${numberText(value)}</td><td class="pred">${numberText(prediction?.pred?.[dimension])}</td><td class="diff">${prediction ? numberText(Math.abs(value - prediction.pred[dimension])) : '—'}</td></tr>`).join('');
  $('states').innerHTML = current.state[frame].map((value, dimension) => `<div class="value"><span>S${dimension + 1}</span><strong>${numberText(value)}</strong></div>`).join('');
  drawCharts();
  if (seek) seekVideos();
}

function prepareCanvas(id) {
  const element = $(id), scale = devicePixelRatio || 1, width = element.clientWidth, height = element.clientHeight;
  if (!width) return null;
  element.width = Math.round(width * scale); element.height = Math.round(height * scale);
  const context = element.getContext('2d'); context.scale(scale, scale); context.clearRect(0, 0, width, height);
  const padding = 18; context.strokeStyle = '#263a50'; context.lineWidth = 1;
  for (let line = 0; line <= 4; line++) { const y = padding + (height - 2 * padding) * line / 4; context.beginPath(); context.moveTo(padding, y); context.lineTo(width - padding, y); context.stroke(); }
  return { context, width, height, padding };
}
function plot(box, values, color, lower, upper) {
  const { context, width, height, padding } = box;
  context.strokeStyle = color; context.lineWidth = 1.8; context.beginPath();
  values.forEach((value, index) => {
    if (!Number.isFinite(value)) return;
    const x = padding + (width - 2 * padding) * index / Math.max(1, values.length - 1);
    const y = height - padding - (value - lower) / (upper - lower) * (height - 2 * padding);
    if (index) context.lineTo(x, y); else context.moveTo(x, y);
  });
  context.stroke();
}
function marker(box) {
  const { context, width, height, padding } = box;
  const x = padding + (width - 2 * padding) * frame / Math.max(1, current.frames - 1);
  context.strokeStyle = '#fff'; context.lineWidth = 1.4; context.beginPath(); context.moveTo(x, padding); context.lineTo(x, height - padding); context.stroke();
}
function drawCharts() {
  if (!current) return;
  const dimension = Number($('dimension').value || 0);
  const groundTruth = current.action.map(action => action[dimension]);
  const predicted = predictions?.frames?.map(row => row.pred[dimension]) || [];
  let box = prepareCanvas('actionChart');
  if (box) {
    const values = groundTruth.concat(predicted).filter(Number.isFinite);
    const lower = Math.min(...values), upper = Math.max(...values), spread = Math.max(upper - lower, 0.0001);
    plot(box, groundTruth, '#5eead4', lower - spread * 0.08, upper + spread * 0.08);
    if (predicted.length === current.frames) plot(box, predicted, '#79a9ff', lower - spread * 0.08, upper + spread * 0.08);
    marker(box);
  }
  box = prepareCanvas('metricChart');
  if (box) {
    if (predictions?.frames?.length === current.frames) {
      const losses = predictions.frames.map(row => row.loss), errors = predictions.frames.map(row => row.mae);
      plot(box, losses, '#fbbf70', 0, Math.max(0.001, ...losses));
      plot(box, errors, '#e9a2db', 0, Math.max(0.001, ...errors));
    }
    marker(box);
  }
}

function tick() {
  if (!playing || !current) return;
  const main = videos[0];
  const currentFrame = Math.floor((main.currentTime - current.videos[cameras[0]].start) * current.fps + 0.02);
  if (currentFrame >= current.frames - 1) { showFrame(current.frames - 1); stop(); return; }
  if (currentFrame >= 0 && currentFrame !== frame) showFrame(currentFrame, false);
  const wrist = videos[1], offset = current.videos[cameras[1]].start - current.videos[cameras[0]].start;
  if (Math.abs(wrist.currentTime - main.currentTime - offset) > 0.15) wrist.currentTime = main.currentTime + offset;
  animation = requestAnimationFrame(tick);
}
async function play() {
  if (!current) return;
  if (frame >= current.frames - 1) showFrame(0);
  seekVideos();
  try { await Promise.all(videos.map(video => video.play())); playing = true; $('play').textContent = 'Ⅱ'; tick(); }
  catch (error) { stop(); console.error(error); }
}

async function choose(id) {
  stop(); current = episodes.find(episode => episode.id === Number(id)); if (!current) return;
  $('episode').value = String(current.id);
  predictions = null; const token = ++requestToken;
  $('task').textContent = current.task; $('episodeLabel').textContent = String(current.id).padStart(3, '0'); $('frameCount').textContent = current.frames; $('seek').max = current.frames - 1;
  $('episodeHint').textContent = `${episodes.length} ready for playback · ${totalEpisodes.toLocaleString()} downloaded`;
  $('modelLabel').textContent = 'Loading…'; $('mode').textContent = 'Loading saved frame-by-frame model inference…'; $('meanLoss').textContent = '—'; $('meanMae').textContent = '—';
  videos.forEach((video, camera) => {
    const info = current.videos[cameras[camera]]; video.poster = info.poster; video.src = info.url; video.playbackRate = Number($('speed').value);
    video.onloadedmetadata = () => { if (current?.id === Number(id)) video.currentTime = info.start; };
  });
  showFrame(0, false);
  try {
    const response = await fetch(`/predictions/episode-${current.id}.json?v=3`);
    if (!response.ok) throw Error('No saved inference for this episode');
    const data = await response.json(); if (token !== requestToken) return;
    if (data.episode !== current.id || data.frames.length !== current.frames || data.frames.some((row, index) => row.frame !== index)) throw Error('Inference frames do not match the episode');
    predictions = data; $('modelLabel').textContent = 'SmolVLA · 24,356 steps'; $('mode').textContent = 'Recorded demonstration with aligned ground-truth and model actions.';
    $('meanLoss').textContent = numberText(data.mean_loss); $('meanMae').textContent = numberText(data.mean_mae); showFrame(frame, false);
  } catch (error) { if (token !== requestToken) return; $('modelLabel').textContent = 'Ground truth only'; $('mode').textContent = error.message; showFrame(frame, false); }
}

$('play').onclick = () => playing ? stop() : play();
$('back').onclick = () => { stop(); showFrame(frame - 1); };
$('forward').onclick = () => { stop(); showFrame(frame + 1); };
$('seek').oninput = event => { stop(); showFrame(Number(event.target.value)); };
$('speed').onchange = event => videos.forEach(video => video.playbackRate = Number(event.target.value));
$('episode').onchange = event => { history.replaceState(null, '', `?episode=${event.target.value}`); choose(event.target.value); };
$('dimension').onchange = drawCharts;
document.addEventListener('keydown', event => { if (['INPUT', 'SELECT', 'BUTTON'].includes(document.activeElement.tagName)) return; if (event.code === 'Space') { event.preventDefault(); playing ? stop() : play(); } if (event.code === 'ArrowLeft') { stop(); showFrame(frame - 1); } if (event.code === 'ArrowRight') { stop(); showFrame(frame + 1); } });
window.addEventListener('resize', drawCharts);
$('dimension').innerHTML = Array.from({ length: 7 }, (_, index) => `<option value="${index}">Action A${index + 1}</option>`).join('');
$('rolloutPlay').onclick = () => {
  const video = $('rolloutVideo');
  if (video.paused) video.play(); else video.pause();
};
$('rolloutVideo').onplay = () => { $('rolloutPlay').textContent = 'Pause rollout'; };
$('rolloutVideo').onpause = () => { $('rolloutPlay').textContent = 'Play rollout'; };
fetch('/rollouts/task7.json?v=1').then(response => {
  if (!response.ok) throw Error('No simulator rollout yet');
  return response.json();
}).then(rollout => {
  if (!rollout.video || !Number.isFinite(rollout.steps) || typeof rollout.success !== 'boolean') throw Error('Incomplete rollout result');
  $('rolloutVideo').src = rollout.video;
  if (rollout.poster) $('rolloutVideo').poster = rollout.poster;
  $('rolloutStatus').textContent = rollout.success ? 'TASK SUCCESS' : 'TASK NOT COMPLETED';
  $('rolloutStatus').style.color = rollout.success ? 'var(--cyan)' : 'var(--amber)';
  const details = $('rolloutDetails');
  details.innerHTML = '<strong>Independent simulator run</strong><dl><dt>Task</dt><dd id="rolloutTask"></dd><dt>Outcome</dt><dd id="rolloutOutcome"></dd><dt>Length</dt><dd id="rolloutLength"></dd><dt>Evaluation</dt><dd id="rolloutNote"></dd></dl>';
  $('rolloutTask').textContent = rollout.task;
  $('rolloutOutcome').textContent = rollout.success ? 'Success' : 'No success within the episode limit';
  $('rolloutLength').textContent = `${rollout.steps} steps at ${rollout.fps} Hz`;
  $('rolloutNote').textContent = rollout.note;
  $('rolloutPanel').hidden = false;
}).catch(error => console.info(error.message));
fetch('/episodes.json?v=3').then(response => { if (!response.ok) throw Error('Prepare the viewer index first'); return response.json(); }).then(data => {
  episodes = data.episodes; totalEpisodes = data.total_episodes || episodes.length;
  const preferred = [53, 347]; episodes.sort((a, b) => (preferred.includes(a.id) ? -1 : 0) - (preferred.includes(b.id) ? -1 : 0) || a.id - b.id);
  $('episode').innerHTML = episodes.map(episode => `<option value="${episode.id}">${preferred.includes(episode.id) ? 'TEST · ' : ''}Episode ${String(episode.id).padStart(3, '0')} · ${episode.duration.toFixed(1)} s</option>`).join('');
  const requested = Number(new URLSearchParams(location.search).get('episode'));
  choose(episodes.some(episode => episode.id === requested) ? requested : episodes[0].id);
}).catch(error => { $('task').textContent = error.message; });
