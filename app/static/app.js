let ws;
let paused = false;
let state = {};

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws/client`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'transcript') {
      appendTranscript(msg.text);
    }
    if (msg.type === 'coach_card') {
      appendCoachCard(msg.card);
    }
    if (msg.type === 'state') {
      state = msg.state;
      paused = !!state.paused;
      syncControls();
    }
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}

function appendTranscript(text) {
  const feed = document.getElementById('transcriptFeed');
  const div = document.createElement('div');
  div.className = 'transcript-line';
  div.textContent = text;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function appendCoachCard(card) {
  const feed = document.getElementById('coachFeed');
  const div = document.createElement('div');
  div.className = 'coach-card';
  const title = document.createElement('h3');
  title.textContent = card.issue || 'Coach response';
  div.appendChild(title);
  const list = document.createElement('ul');
  (card.bullets || []).slice(0, 6).forEach((b) => {
    const li = document.createElement('li');
    li.textContent = b;
    list.appendChild(li);
  });
  div.appendChild(list);
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function sendTranscript(text) {
  if (!text.trim() || state.do_not_listen) return;
  ws?.send(JSON.stringify({ type: 'transcript', text }));
}

function manualPush() {
  ws?.send(JSON.stringify({ type: 'manual_push' }));
}

function togglePause() {
  paused = !paused;
  ws?.send(JSON.stringify({ type: 'toggle_pause', paused }));
  document.getElementById('pauseBtn').textContent = paused ? 'Resume' : 'Pause Listening';
}

function syncControls() {
  document.getElementById('autoMode').checked = state.mode === 'auto';
  document.getElementById('pauseBtn').textContent = state.paused ? 'Resume' : 'Pause Listening';
  document.getElementById('doNotListen').checked = !!state.do_not_listen;
}

window.addEventListener('DOMContentLoaded', () => {
  connectWS();

  document.getElementById('manualInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const text = e.target.value;
      e.target.value = '';
      appendTranscript(text);
      sendTranscript(text);
    }
  });

  document.getElementById('manualPush').addEventListener('click', manualPush);
  document.getElementById('pauseBtn').addEventListener('click', togglePause);

  document.getElementById('doNotListen').addEventListener('change', (e) => {
    ws?.send(
      JSON.stringify({
        type: 'set_do_not_listen',
        do_not_listen: e.target.checked,
      })
    );
  });

  document.getElementById('autoMode').addEventListener('change', (e) => {
    const mode = e.target.checked ? 'auto' : 'manual';
    ws?.send(
      JSON.stringify({
        type: 'set_mode',
        mode,
      })
    );
  });

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      manualPush();
    }
    if (e.ctrlKey && (e.key.toLowerCase() === 'p')) {
      e.preventDefault();
      togglePause();
    }
  });
});
