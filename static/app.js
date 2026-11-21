const canvas = document.querySelector('#drawing');
const ctx = canvas.getContext('2d');
const placeholder = document.querySelector('#placeholder');
const recognize = document.querySelector('#recognize');
const errorBox = document.querySelector('#error');
let drawing = false;
let hasInk = false;

ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#20231f';
ctx.lineWidth = 27;

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height };
}
canvas.addEventListener('pointerdown', event => {
  drawing = true;
  canvas.setPointerCapture(event.pointerId);
  const p = point(event);
  ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + .1, p.y + .1); ctx.stroke();
  hasInk = true; placeholder.classList.add('hide'); errorBox.textContent = '';
});
canvas.addEventListener('pointermove', event => {
  if (!drawing) return;
  const p = point(event); ctx.lineTo(p.x, p.y); ctx.stroke();
});
function stopDrawing() { drawing = false; }
canvas.addEventListener('pointerup', stopDrawing);
canvas.addEventListener('pointercancel', stopDrawing);

document.querySelector('#clear').addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hasInk = false; placeholder.classList.remove('hide'); errorBox.textContent = '';
  document.querySelector('#resultEmpty').hidden = false;
  document.querySelector('#resultContent').hidden = true;
});

recognize.addEventListener('click', async () => {
  if (!hasInk) { errorBox.textContent = 'Draw a digit before asking the model.'; return; }
  recognize.disabled = true; recognize.innerHTML = 'Thinking <span>…</span>'; errorBox.textContent = '';
  try {
    const response = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({image: canvas.toDataURL('image/png')}) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Prediction failed.');
    const percent = Math.round(result.confidence * 100);
    document.querySelector('#digit').textContent = result.digit;
    document.querySelector('#confidenceText').textContent = `${percent}%`;
    document.querySelector('#confidenceBar').style.width = `${percent}%`;
    document.querySelector('#runnerUp').textContent = `Next guess: ${result.second_digit} (${Math.round(result.second_confidence * 100)}%)`;
    document.querySelector('#resultEmpty').hidden = true;
    document.querySelector('#resultContent').hidden = false;
  } catch (error) { errorBox.textContent = error.message || 'Could not reach the model.'; }
  finally { recognize.disabled = false; recognize.innerHTML = 'Recognize <span>→</span>'; }
});
