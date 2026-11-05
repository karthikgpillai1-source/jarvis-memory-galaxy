/* J.A.R.V.I.S. COMM-LINK - local GGUF chatbot client with memory-galaxy tools.
   Runs after app.js (module order). Talks to same-origin /api/chat (server.py
   serves it from a local GGUF via llama.cpp, injects recalled memories,
   handles store/focus tools). No Ollama, no cloud model. */
const $ = id => document.getElementById(id);
const chatwin = $('chatwin'), msgs = $('chat-msgs'), form = $('chat-form'),
      input = $('chat-input'), dot = $('chat-dot'), statusEl = $('chat-status'),
      modelSel = $('model-select');

let history = [];
let model = localStorage.getItem('jarvis_model_v2') || '';
let busy = false;

const SECTOR_CSS = { personal: '#00d4ff', work: '#00ff9d', ideas: '#ffb300', important: '#ff3b5c' };

/* ---------- window chrome ---------- */
function openChat() { chatwin.classList.remove('closed'); $('btn-chat')?.classList.add('active'); setTimeout(() => input.focus(), 80); }
function closeChat() { chatwin.classList.add('closed'); $('btn-chat')?.classList.remove('active'); }
function toggleChat() { chatwin.classList.contains('closed') ? openChat() : closeChat(); }
$('btn-chat')?.addEventListener('click', toggleChat);
$('chat-close').onclick = closeChat;
addEventListener('keydown', e => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '');
  if ((e.key.toLowerCase() === 'c' || e.key === '6') && !typing) toggleChat();
  if (e.key === 'Escape' && !chatwin.classList.contains('closed') && document.activeElement !== input) closeChat();
});
document.querySelectorAll('.chat-chips button').forEach(b => b.onclick = () => send(b.dataset.q));

/* ---------- messages ---------- */
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function pushMsg(kind, text, recalled = null) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + kind;
  const who = kind === 'user' ? 'YOU' : kind === 'err' ? 'UPLINK ERROR' : 'J.A.R.V.I.S.';
  let html = `<div class="msg-who mono">${who}</div><div class="msg-body"></div>`;
  wrap.innerHTML = html;
  wrap.querySelector('.msg-body').textContent = text;
  if (recalled?.length) {
    const src = document.createElement('div');
    src.className = 'msg-src mono';
    src.innerHTML = '◈ RECALLED: ' + recalled.map(r =>
      `<button data-mid="${esc(r.id)}" title="Focus this memory (score ${r.score})" style="--c:${SECTOR_CSS[r.sector] || '#00d4ff'}">◆ ${esc(r.title)}</button>`
    ).join(' ');
    wrap.appendChild(src);
    src.querySelectorAll('button').forEach(b => b.onclick = () => window.jarvis?.selectMemory(b.dataset.mid, true));
  }
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;
  return wrap;
}
function typeInto(el, text) {
  return new Promise(res => {
    let i = 0;
    const body = el.querySelector('.msg-body');
    body.textContent = '';
    (function tick() {
      i += 3;
      body.textContent = text.slice(0, i);
      msgs.scrollTop = msgs.scrollHeight;
      i < text.length ? requestAnimationFrame(tick) : res();
    })();
  });
}
let typingEl = null;
function showTyping() {
  typingEl = document.createElement('div');
  typingEl.className = 'msg jarvis typing';
  typingEl.innerHTML = '<div class="msg-who mono">J.A.R.V.I.S.</div><div class="tdots"><span></span><span></span><span></span></div>';
  msgs.appendChild(typingEl); msgs.scrollTop = msgs.scrollHeight;
}
function hideTyping() { typingEl?.remove(); typingEl = null; }

/* ---------- backend ---------- */
const FRONTEND_BUILD = 18; // must match server.py BUILD — mismatch = stale backend
async function refreshStatus() {
  try {
    const r = await fetch('/api/models');
    const data = await r.json();
    const tools = (data.models || []).filter(m => m.tools && !m.name.includes('embed'));
    const all = (data.models || []).filter(m => !m.name.includes('embed'));
    const list = tools.length ? tools : all;
    modelSel.innerHTML = '';
    list.forEach(m => {
      const o = document.createElement('option');
      o.value = m.name; o.textContent = m.name;
      modelSel.appendChild(o);
    });
    if (!list.length) throw new Error('no models');
    if (!model || ![...modelSel.options].some(o => o.value === model)) {
      model = data.default && [...modelSel.options].some(o => o.value === data.default)
        ? data.default : modelSel.options[0].value;
    }
    modelSel.value = model;
    const galaxy = window.jarvis?.getMemories ? 'GALAXY LINKED' : 'GALAXY OFFLINE';
    // BUILD 18: llama.cpp is the only engine, so state it plainly.
    dot.className = 'chat-dot on';
    statusEl.textContent = `LOCAL GGUF · ${list.length} MODELS · ${galaxy}`;
    try {
      const h = await fetch('/api/health');
      const hd = await h.json();
      if (hd && typeof hd.build === 'number' && hd.build < FRONTEND_BUILD) {
        dot.className = 'chat-dot off';
        statusEl.textContent = `BACKEND STALE (build ${hd.build} < ${FRONTEND_BUILD}) — RESTART JARVIS.bat`;
      } else if (hd && hd.build === FRONTEND_BUILD) {
        const bits = [];
        if (hd.ocr && !hd.ocr.available) bits.push('OCR OFF');
        if (hd.browser && !hd.browser.available) bits.push('CDP OFF');
        if (hd.code && hd.code.auto) bits.push('CODE AUTO');
        // BUILD 18: no fallback exists, so surface real errors instead.
        const inf = hd.inference;
        if (inf && inf.errors) bits.push(`${inf.errors} ERR`);
        if (inf && inf.last_error) bits.push(inf.last_error.slice(0, 48));
        if (bits.length) statusEl.textContent += ` · ${bits.join(' ')}`;
      }
    } catch (e) {}
  } catch (e) {
    dot.className = 'chat-dot off'; statusEl.textContent = 'OFFLINE — START BACKEND';
  }
}
modelSel.onchange = () => { model = modelSel.value; localStorage.setItem('jarvis_model_v2', model); };

let osSid = null, osCard = null, osTimer = null;

function clearOsTimer() { if (osTimer) { clearTimeout(osTimer); osTimer = null; } }

async function handleFinal(data) {
  const bridge = window.jarvis;
  const ready = bridge && ['storeMemory', 'editMemory', 'deleteMemory', 'selectMemory', 'setFilter', 'command'].every(f => typeof bridge[f] === 'function');
  if (!ready) {
    pushMsg('err', 'Galaxy link offline, sir — the 3D core module did not initialize. Hard-refresh with Ctrl+Shift+R; if it persists, press F12 and read me the red console error. I can still answer from memory.');
  } else {
    for (const a of data.actions || []) {
      try {
        if (a.type === 'store_memory') bridge.storeMemory(a.memory);
        else if (a.type === 'edit_memory') bridge.editMemory(a.id, a.patch);
        else if (a.type === 'delete_memory') bridge.deleteMemory(a.id);
        else if (a.type === 'focus_memory') bridge.selectMemory(a.id, true);
        else if (a.type === 'set_filter') bridge.setFilter(a.sector);
        else if (a.type === 'galaxy_command') bridge.command(a.command);
        else pushMsg('err', `Unknown action '${a.type}' — core ignored it, sir.`);
      } catch (err) {
        pushMsg('err', `Action '${a.type}' failed in the 3D core: ${err?.message || err}`);
      }
    }
  }
  history.push({ role: 'assistant', content: data.reply });
  history = history.slice(-24);
  const el = pushMsg('jarvis', '', data.recalled);
  window.jarvisSpeak?.(data.reply);
  await typeInto(el, data.reply);
  try { if (converseOn) converseAfterTurn(); } catch (e) {}
  if (data.shot) {
    // "See what he sees" — annotated boxes first, plain shot as fallback.
    try {
      const shot = document.createElement('img');
      shot.className = 'shot-thumb';
      shot.alt = 'What JARVIS sees';
      shot.title = 'Latest screenshot (numbered boxes) — click to open full size';
      shot.loading = 'lazy';
      shot.src = '/api/shot_annotated?ts=' + Date.now();
      shot.onclick = () => window.open(shot.src, '_blank');
      shot.onerror = () => {
        try { shot.src = '/api/shot?ts=' + Date.now(); } catch (e) { shot.remove(); }
      };
      el.appendChild(shot);
      msgs.scrollTop = msgs.scrollHeight;
    } catch (e) {}
  }
}

function renderOsCard(p) {
  clearOsTimer();
  osSid = p.sid;
  syncOsKill();
  if (osCard) osCard.remove();
  const wrap = document.createElement('div');
  wrap.className = 'msg jarvis os-card';
  const steps = (p.steps || []).map(s => `<div class="os-step">${s}</div>`).join('');
  wrap.innerHTML = `<div class="msg-who mono">J.A.R.V.I.S. · SCREEN CONTROL</div>
    <div class="os-steps">${steps || '<div class="os-step dim">Task started…</div>'}</div>
    <div class="os-proposed">◈ PROPOSED: <b></b></div>
    <div class="os-actions">
      <button class="os-yes">✔ APPROVE</button>
      <button class="os-no">✕ DENY</button>
      <button class="os-stop">⏹ STOP ALL</button>
    </div>
    <div class="os-note mono">Auto-denies in 120s · mouse to screen corner aborts motion</div>`;
  wrap.querySelector('.os-proposed b').textContent = p.human;
  const [yes, no, stop] = wrap.querySelectorAll('button');
  yes.onclick = () => osRespond(true, wrap, yes);
  no.onclick = () => osRespond(false, wrap, no);
  stop.onclick = () => osKill(true);
  msgs.appendChild(wrap); msgs.scrollTop = msgs.scrollHeight;
  osCard = wrap;
  osTimer = setTimeout(() => { if (osCard === wrap) osRespond(false, wrap, no, true); }, 120000);
}

async function osRespond(approved, wrap, btn, auto = false) {
  if (!osSid) return;
  clearOsTimer();
  wrap.querySelectorAll('button').forEach(b => b.disabled = true);
  wrap.querySelector('.os-proposed').innerHTML = `◈ <span class="os-working">${auto ? 'Timed out — denied.' : approved ? 'Approved — executing…' : 'Denied.'}</span>`;
  try {
    const r = await fetch('/api/os_approve', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sid: osSid, approved }),
    });
    const data = await r.json();
    if (data.error) { pushMsg('err', 'Screen control halted, sir: ' + data.error); osDone(wrap, false); return; }
    if (data.pending_approval) { renderOsCard({ ...data.pending_approval, steps: data.pending_approval.steps }); }
    else { osDone(wrap, true); await handleFinal(data); }
  } catch (e) {
    pushMsg('err', 'Screen control link failed, sir. Is server.py still running?');
    osDone(wrap, false);
  }
}

function osDone(wrap, ok) {
  clearOsTimer();
  osSid = null; syncOsKill();
  const badge = document.createElement('div');
  badge.className = 'os-step ' + (ok ? 'ok' : 'stopped');
  badge.textContent = ok ? '■ Task complete.' : '■ Task stopped.';
  wrap.querySelector('.os-steps').appendChild(badge);
  wrap.querySelector('.os-actions')?.remove();
  wrap.querySelector('.os-proposed')?.remove();
  msgs.scrollTop = msgs.scrollHeight;
  if (osCard === wrap) osCard = null;
}

async function osKill(fromCard = false) {
  const sid = osSid;
  clearOsTimer();
  osSid = null; syncOsKill();
  if (sid) {
    try { await fetch('/api/os_kill', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sid }) }); } catch (e) {}
  }
  if (osCard) { osDone(osCard, false); }
  if (!fromCard) pushMsg('jarvis', 'Screen control halted, sir. All pending actions cancelled.');
}

function syncOsKill() { const b = $('btn-oskill'); if (b) b.classList.toggle('on', !!osSid); }
$('btn-oskill')?.addEventListener('click', () => osKill(false));

async function send(text) {
  text = (text || '').trim();
  if (!text || busy) return;
  openChat();
  busy = true;
  pushMsg('user', text);
  history.push({ role: 'user', content: text });
  input.value = '';
  showTyping();
  try {
    const memories = window.jarvis ? window.jarvis.getMemories() : [];
    const ui_state = window.jarvis ? window.jarvis.getState() : {};
    const r = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: history.slice(-12), memories, ui_state, model }),
    });
    const data = await r.json();
    hideTyping();
    if (data.error) { pushMsg('err', 'Uplink failed, sir: ' + data.error); }
    else if (data.pending_approval) { renderOsCard(data.pending_approval); }
    else { await handleFinal(data); }
  } catch (e) {
    hideTyping();
    pushMsg('err', 'Cannot reach the backend, sir. Ensure server.py is running on this host.');
  }
  busy = false;
}
form.addEventListener('submit', e => { e.preventDefault(); send(input.value); });
window.askJarvis = send;

/* ================= SPEECH-TO-SPEECH (push-to-talk + always-listening wake word) =================
   Hybrid STT: browser Web Speech API first (instant, no installs — Chrome/Edge),
   MediaRecorder -> POST /api/transcribe (local Whisper) as fallback for
   Firefox / repeated browser-STT failures / offline use.
   Wake loop: continuous low-cost listener for "hey JARVIS" -> single-shot
   command listen -> existing send() -> spoken reply (auto voice-on) -> back to wake.
   Barge-in: hearing the wake word (or tapping mic) while Jarvis speaks stops TTS. */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition || null;
const micBtn = $('btn-mic'), wakeBtn = $('btn-wake'), voiceStatus = $('voice-status');
const STT_LANG = (navigator.language && /^en/i.test(navigator.language)) ? navigator.language : 'en-IN';
const WAKE_RE = /\bhey[ ,-]*jarvis\b|\bok[ ,-]*jarvis\b|\bjarvis\b/i;

let sttMode = 'idle'; // idle | wake | command | recording
let wakeOn = true; // wake-word daemon always on — no talk button needed
let converseOn = false; // transient awake state (not persisted)
let wakeBlocked = false, wakeStarted = false;
let wakeRec = null, cmdRec = null, wakeRestartTimer = null, cmdSilenceTimer = null;
let sttFails = 0, lastWakeAt = 0, mediaStream = null, mediaRec = null, mediaChunks = [], mediaStopTimer = null;
let lastActivity = Date.now();
const WAKE_SLEEP_MS = 30000; // auto-sleep back to wake-only after 30s silence

function setStatus(text, cls = '') {
  if (!voiceStatus) return;
  voiceStatus.textContent = text;
  voiceStatus.className = 'mono ' + cls;
}
function syncSpeechUI() {
  if (micBtn) {
    micBtn.classList.toggle('listening', sttMode === 'command' || sttMode === 'recording');
    micBtn.classList.toggle('wake-active', wakeOn && sttMode === 'wake');
    micBtn.textContent = sttMode === 'command' || sttMode === 'recording' ? '⏹' : '🎤';
    micBtn.title = !SR && sttMode === 'idle'
      ? 'Speak — offline record (this browser has no Web Speech API)'
      : sttMode === 'command' ? 'Listening… click to cancel'
      : sttMode === 'recording' ? 'Recording… click to stop & transcribe'
      : 'Speak — click and talk';
  }
  if (wakeBtn) {
    // ◉ is a status indicator only — never a talk button.
    const awake = !!converseOn;
    const sleeping = wakeOn && !awake && !wakeBlocked;
    wakeBtn.classList.toggle('on', awake || sleeping);
    wakeBtn.title = wakeBlocked
      ? 'Mic blocked — click once to enable ears'
      : awake ? 'JARVIS awake — just talk'
      : sleeping ? 'JARVIS sleeping — say HEY JARVIS'
      : 'Voice starting…';
    wakeBtn.style.opacity = '1';
  }
  if (micBtn) micBtn.classList.toggle('wake-active', !!converseOn);
}
function touchActivity() { lastActivity = Date.now(); }
function autoVoiceOn() { try { window.jarvisVoice?.enable(); } catch (e) {} }
function stopRec(rec) { try { rec && rec.stop(); } catch (e) {} try { rec && rec.abort && rec.abort(); } catch (e) {} }

function scheduleWakeRestart(ms = 350) {
  clearTimeout(wakeRestartTimer);
  if (!wakeOn || sttMode !== 'wake' || !SR) return;
  wakeRestartTimer = setTimeout(() => {
    if (!wakeOn || sttMode !== 'wake') return;
    try { wakeRec && wakeRec.start(); } catch (e) {}
  }, ms);
}

function startWake() {
  if (converseOn) return true; // awake — wake daemon stays parked
  if (!SR) {
    wakeBlocked = false;
    setStatus('WAKE NEEDS CHROME/EDGE — TYPE BELOW, EARS OFFLINE', 'err');
    syncSpeechUI();
    return false;
  }
  stopRec(cmdRec);
  sttMode = 'wake';
  if (!wakeRec) {
    wakeRec = new SR();
    wakeRec.continuous = true;
    wakeRec.interimResults = false;
    wakeRec.lang = STT_LANG;
    wakeRec.onresult = (e) => {
      try {
        const res = e.results[e.results.length - 1];
        const transcript = (res[0]?.transcript || '').trim();
        if (!transcript || !WAKE_RE.test(transcript)) return;
        const now = Date.now();
        if (now - lastWakeAt < 1500) return; // debounce
        lastWakeAt = now;
        onWakeHeard(transcript);
      } catch (err) {}
    };
    wakeRec.onerror = (e) => {
      const err = e?.error || '';
      if (err === 'not-allowed' || err === 'service-not-allowed') {
        wakeBlocked = true;
        sttMode = 'idle';
        syncSpeechUI();
        setStatus('MIC BLOCKED — CLICK ONCE TO ENABLE EARS', 'err');
        pushMsg('err', 'Microphone blocked, sir — click the lock icon in the address bar to allow the mic, then click once anywhere and I will start listening for HEY JARVIS.');
        return;
      }
      if (err === 'network') {
        sttFails++;
        setStatus('WAKE NETWORK ERROR — RETRYING…', 'err');
      }
      scheduleWakeRestart(900);
    };
    wakeRec.onend = () => scheduleWakeRestart(300); // Chrome auto-stops ~60s
  }
  try {
    wakeRec.start();
    wakeBlocked = false; wakeStarted = true; touchActivity();
    setStatus(converseOn ? '● LISTENING — JUST SPEAK, SIR' : "SLEEPING — SAY 'HEY JARVIS'", 'live');
  } catch (e) { scheduleWakeRestart(500); }
  syncSpeechUI();
  return true;
}
function stopWakeToIdle() {
  clearTimeout(wakeRestartTimer);
  stopRec(wakeRec);
  if (sttMode === 'wake') sttMode = 'idle';
  syncSpeechUI();
  setStatus("SLEEPING — SAY 'HEY JARVIS' (OR TYPE BELOW)", '');
}

function onWakeHeard(transcript) {
  // Wake word is only the doorbell — the full hands-free loop takes it from here.
  autoVoiceOn();
  try { window.jarvisStopSpeak?.(); } catch (e) {} // barge-in
  touchActivity();
  setStatus(`YES SIR? — WAKING…`, 'live');
  try { clearTimeout(wakeRestartTimer); } catch (e) {}
  try { stopRec(wakeRec); } catch (e) {}
  sttMode = 'idle';
  converseStart({ fromWake: true, heard: transcript });
}

function listenOnce({ fromWake = false } = {}) {
  if (!SR) { whisperRecordOnce(); return; }
  if (sttMode === 'command') return;
  clearTimeout(wakeRestartTimer);
  stopRec(wakeRec);
  try { window.jarvisStopSpeak?.(); } catch (e) {} // barge-in on mic tap
  sttMode = 'command';
  syncSpeechUI();
  setStatus(fromWake ? 'LISTENING… SPEAK NOW, SIR' : 'LISTENING… SPEAK NOW (CLICK ⏹ TO CANCEL)', 'live');
  openChat();
  cmdRec = new SR();
  cmdRec.continuous = false;
  cmdRec.interimResults = true;
  cmdRec.maxAlternatives = 1;
  cmdRec.lang = STT_LANG;
  let finalText = '', gotFinal = false;
  const armSilence = () => {
    clearTimeout(cmdSilenceTimer);
    cmdSilenceTimer = setTimeout(() => {
      if (sttMode === 'command' && !gotFinal) { try { cmdRec.stop(); } catch (e) {} }
    }, 8000);
  };
  armSilence();
  cmdRec.onresult = (e) => {
    armSilence();
    let interim = '';
    for (const res of e.results) {
      const t = res[0]?.transcript || '';
      if (res.isFinal) { finalText += t + ' '; gotFinal = true; }
      else interim += t;
    }
    const show = (finalText + interim).trim();
    if (show) setStatus(`HEARING: "${show.slice(0, 90)}"`, 'live');
  };
  cmdRec.onerror = (e) => {
    clearTimeout(cmdSilenceTimer);
    const err = e?.error || '';
    if (err === 'not-allowed' || err === 'service-not-allowed') {
      sttMode = wakeOn ? 'wake' : 'idle';
      setStatus('MIC BLOCKED — CLICK ONCE TO ENABLE EARS', 'err');
      pushMsg('err', 'Microphone blocked, sir — allow it in the address-bar lock icon, then click once anywhere.');
    } else if (err === 'aborted') {
      // user cancelled — fall through to onend handling
    } else {
      sttFails++;
      setStatus(`DIDN'T CATCH THAT (${err || 'no speech'}) — TRY AGAIN`, 'err');
    }
    syncSpeechUI();
  };
  cmdRec.onend = async () => {
    clearTimeout(cmdSilenceTimer);
    const text = (finalText || '').trim();
    cmdRec = null;
    if (!text) {
      sttMode = wakeOn ? 'wake' : 'idle';
      if (wakeOn) startWake(); else { syncSpeechUI(); setStatus("SLEEPING — SAY 'HEY JARVIS' (OR TYPE BELOW)", ''); }
      if (sttFails >= 3 && !fromWake) {
        pushMsg('err', 'Browser speech keeps failing, sir — my offline ears take over automatically from here.');
      }
      return;
    }
    sttFails = 0;
    await handleVoiceText(text);
  };
  try { cmdRec.start(); } catch (e) {
    sttMode = wakeOn ? 'wake' : 'idle';
    syncSpeechUI();
  }
}

async function handleVoiceText(text) {
  text = (text || '').trim().slice(0, 2000);
  if (!text) return;
  autoVoiceOn();
  touchActivity();
  sttMode = 'idle';
  syncSpeechUI();
  setStatus(`HEARD: "${text.slice(0, 90)}" — WORKING…`, 'heard');
  if (busy) {
    if (converseOn) {
      // Hands-free queue: hold one utterance, deliver after this turn.
      pendingVoice = text;
      setStatus('NOTED — FINISHING CURRENT TASK…', 'heard');
      return;
    }
    setStatus('JARVIS IS BUSY — ONE SEC, SIR…', 'err');
    pushMsg('err', 'Easy, sir — let me finish the last task first, then speak again.');
    if (wakeOn && !converseOn) startWake();
    return;
  }
  try {
    await send(text);
  } finally {
    touchActivity();
    if (converseOn) converseAfterTurn();
    else if (wakeOn) startWake();
    else { syncSpeechUI(); setStatus("SLEEPING — SAY 'HEY JARVIS' (OR TYPE BELOW)", ''); }
  }
}

/* ---- Whisper fallback: record -> POST /api/transcribe ---- */
async function whisperRecordOnce() {
  if (sttMode === 'recording') { stopWhisperRecord(); return; } // second tap stops
  try {
    if (!mediaStream) mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus('MIC BLOCKED — ALLOW MICROPHONE, THEN RETRY', 'err');
    pushMsg('err', 'Microphone blocked, sir — allow it in the address-bar lock icon and try again.');
    return;
  }
  try { window.jarvisStopSpeak?.(); } catch (e) {}
  clearTimeout(wakeRestartTimer);
  stopRec(wakeRec);
  sttMode = 'recording';
  syncSpeechUI();
  openChat();
  autoVoiceOn();
  setStatus('RECORDING… SPEAK NOW (TAP 🎤 TO STOP, 15S MAX)', 'live');
  mediaChunks = [];
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : '';
  mediaRec = mime ? new MediaRecorder(mediaStream, { mimeType: mime }) : new MediaRecorder(mediaStream);
  mediaRec.ondataavailable = (e) => { if (e.data?.size) mediaChunks.push(e.data); };
  mediaRec.onstop = whisperUpload;
  try { mediaRec.start(250); } catch (e) {
    sttMode = wakeOn ? 'wake' : 'idle';
    syncSpeechUI();
    return;
  }
  clearTimeout(mediaStopTimer);
  mediaStopTimer = setTimeout(stopWhisperRecord, 15000);
}
function stopWhisperRecord() {
  clearTimeout(mediaStopTimer);
  try { mediaRec && mediaRec.state !== 'inactive' && mediaRec.stop(); } catch (e) { whisperUpload(); }
}
async function whisperUpload() {
  clearTimeout(mediaStopTimer);
  const rec = mediaRec; mediaRec = null;
  void rec;
  const blob = new Blob(mediaChunks, { type: 'audio/webm' });
  mediaChunks = [];
  if (!blob.size) {
    sttMode = wakeOn ? 'wake' : 'idle';
    if (wakeOn) startWake(); else { syncSpeechUI(); setStatus('NOTHING RECORDED — TRY AGAIN', 'err'); }
    return;
  }
  setStatus('TRANSCRIBING OFFLINE (WHISPER)…', 'live');
  syncSpeechUI();
  try {
    const fd = new FormData();
    fd.append('audio', blob, 'cmd.webm');
    const r = await fetch('/api/transcribe', { method: 'POST', body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || ('transcribe ' + r.status));
    const text = (data.text || '').trim();
    if (!text) throw new Error('empty transcript');
    await handleVoiceText(text);
  } catch (e) {
    sttMode = wakeOn ? 'wake' : 'idle';
    if (wakeOn) startWake(); else syncSpeechUI();
    setStatus('TRANSCRIBE FAILED — ' + String(e.message || e).slice(0, 80).toUpperCase(), 'err');
    pushMsg('err', 'Offline transcribe failed, sir: ' + (e.message || e) + ' — backend needs: pip install faster-whisper (+ ffmpeg on PATH). Browser mic (Chrome/Edge) still works without it.');
  }
}

/* ================= WAKE-WORD + CONVERSATION (no talk button) =================
   Dormant: cheap continuous listener for "hey JARVIS" / "JARVIS" only.
   Awake: full hands-free loop LISTENING → HEARING → THINKING → SPEAKING.
   Auto-sleep: 30s silence awake → back to dormant wake-only.
   ◉ is a status indicator only, never a talk control. */
// one-time migration: old prefs retired, wake daemon is always on
try { localStorage.removeItem('jarvis_wake'); } catch (e) {}
try { localStorage.removeItem('jarvis_converse'); } catch (e) {}
wakeOn = true; converseOn = false;
let cvState = 'off'; // off|listening|hearing|thinking|speaking
let cvRec = null, cvRecActive = false, cvEpoch = 0;
let cvAcc = '', cvLastResultAt = 0;
let cvCtx = null, cvAnalyser = null, cvFloat = null, cvVadTimer = null;
let cvNoise = 0.008, cvSilMs = 0, cvUtterMs = 0, cvBargeMs = 0;
let cvMR = null, cvChunks = [], cvMRActive = false;
let cvSpeakTimer = null, pendingVoice = null;
// Patience-tuned endpointing: never cut mid-sentence. A turn ends only on
// 1.8s silence AFTER a final transcript, or 3.5s silence on interim-only audio.
const CV_END_SILENCE = 1800, CV_END_INTERIM = 3500;
const CV_MIN_UTTER = 700, CV_MAX_UTTER = 20000, CV_TICK = 120;
let cvInterim = '', cvGotFinal = false, cvPeakHeard = 0;
let cvEmptyStreak = 0, cvForceWhisper = false; // session auto-fallback SR→Whisper

function cvSpeechThr() { return Math.max(0.022, cvNoise * 2.5); }
function cvBargeThr() { return Math.max(0.055, cvNoise * 5); }

function cvRms() {
  if (!cvAnalyser || !cvFloat) return 0;
  try {
    cvAnalyser.getFloatTimeDomainData(cvFloat);
    let s = 0;
    for (let i = 0; i < cvFloat.length; i++) s += cvFloat[i] * cvFloat[i];
    return Math.sqrt(s / cvFloat.length);
  } catch (e) { return 0; }
}

async function cvEnsureMic() {
  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }
  if (!cvCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    cvCtx = new AC();
    const src = cvCtx.createMediaStreamSource(mediaStream);
    cvAnalyser = cvCtx.createAnalyser();
    cvAnalyser.fftSize = 1024;
    cvFloat = new Float32Array(cvAnalyser.fftSize);
    src.connect(cvAnalyser);
  }
  if (cvCtx.state === 'suspended') { try { await cvCtx.resume(); } catch (e) {} }
  // Calibrate ambient noise (~0.8s).
  let peak = 0;
  for (let i = 0; i < 7; i++) {
    await new Promise(r => setTimeout(r, 115));
    peak = Math.max(peak, cvRms());
  }
  cvNoise = Math.min(0.03, Math.max(0.004, peak * 1.4));
}

function cvStopSR() {
  cvRecActive = false;
  try { cvRec && cvRec.stop(); } catch (e) {}
  try { cvRec && cvRec.abort && cvRec.abort(); } catch (e) {}
  cvRec = null;
}

function cvStartSR() {
  if (!SR || !converseOn || cvForceWhisper) return;
  if (cvRecActive) return;
  cvStopSR();
  try {
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    rec.lang = STT_LANG;
    const epoch = cvEpoch;
    rec.onresult = (e) => {
      if (epoch !== cvEpoch || !converseOn) return;
      if (cvState !== 'listening' && cvState !== 'hearing' && cvState !== 'thinking') return;
      // NOTE: e.results is cumulative — only process NEW results since resultIndex,
      // otherwise finals get appended again every event (garbled repeats).
      let interim = '';
      for (let i = e.resultIndex || 0; i < e.results.length; i++) {
        const res = e.results[i];
        const t = res[0]?.transcript || '';
        if (res.isFinal) { cvAcc += t + ' '; cvGotFinal = true; }
        else interim += t;
      }
      cvInterim = interim;
      cvLastResultAt = Date.now();
      touchActivity();
      if ((cvAcc + cvInterim).trim() && cvState === 'listening') {
        cvState = 'hearing'; cvUtterMs = CV_MIN_UTTER; // SR heard voice; count as utterance
        setStatus(`◉ HEARING YOU… "${(cvAcc + cvInterim).trim().slice(0, 80)}"`, 'live');
        syncSpeechUI();
      } else if (cvState === 'hearing' && (cvAcc + cvInterim).trim()) {
        setStatus(`◉ HEARING YOU… "${(cvAcc + cvInterim).trim().slice(0, 80)}"`, 'live');
      }
    };
    rec.onerror = (e) => {
      const err = e?.error || '';
      if (err === 'not-allowed' || err === 'service-not-allowed') {
        wakeBlocked = true;
        sleepBackToWake(true);
        setStatus('MIC BLOCKED — CLICK ONCE TO ENABLE EARS', 'err');
        pushMsg('err', 'Microphone blocked, sir — allow it in the address-bar lock icon, then click once anywhere.');
        return;
      }
      // network/no-speech/aborted: loop below restarts via onend
    };
    rec.onend = () => {
      cvRecActive = false;
      if (!converseOn) return;
      if (cvState === 'listening' || cvState === 'hearing' || cvState === 'thinking') {
        setTimeout(() => { if (converseOn && !cvRecActive &&
          (cvState === 'listening' || cvState === 'hearing' || cvState === 'thinking')) cvStartSR(); }, 350);
      }
    };
    cvRec = rec;
    rec.start();
    cvRecActive = true;
  } catch (e) { cvRecActive = false; }
}

function cvTick() {
  if (!converseOn || cvState === 'off') return;
  const rms = cvRms();
  const now = Date.now();
  // Barge-in: sustained loud energy while he speaks cuts him off.
  if (cvState === 'speaking') {
    if (rms > cvBargeThr()) {
      cvBargeMs += CV_TICK;
      if (cvBargeMs >= 350) {
        cvBargeMs = 0;
        try { window.jarvisStopSpeak?.(); } catch (e) {}
        // speak-end event resumes listening; epoch bump drops stale results
        cvEpoch++;
      }
    } else cvBargeMs = 0;
    return;
  }
  if (cvState === 'thinking') return; // SR keeps captions; endpoint via timer below
  const speakingLoud = rms > cvSpeechThr();
  if (cvState === 'listening') {
    if (speakingLoud) {
      cvState = 'hearing'; cvUtterMs = 0; cvSilMs = 0;
      setStatus('◉ HEARING YOU…', 'live');
      syncSpeechUI();
      if (!SR || cvForceWhisper) cvStartRecord(); // whisper path records VAD-gated audio
    }
    return;
  }
  if (rms > cvSpeechThr()) cvPeakHeard = Math.max(cvPeakHeard, rms);
  if (cvState === 'hearing') {
    if (speakingLoud) { cvUtterMs += CV_TICK; cvSilMs = 0; }
    else cvSilMs += CV_TICK;
    if (cvUtterMs > CV_MAX_UTTER) { cvEndpoint(true); return; }
    if (!SR || cvForceWhisper) {
      // Whisper path: endpoint purely on VAD silence.
      if (cvSilMs >= CV_END_SILENCE && cvUtterMs >= CV_MIN_UTTER) cvStopRecord();
      return;
    }
    // SR path: patient endpointing. Finals mean real words — 1.8s of quiet
    // after them means he finished. Interim-only audio gets 3.5s (slow talkers,
    // pauses mid-sentence must NOT cut). Nothing at all for 6s: false alarm.
    const heard = (cvAcc + cvInterim).trim();
    const quiet = now - cvLastResultAt;
    if (heard && ((cvGotFinal && quiet >= CV_END_SILENCE) || quiet >= CV_END_INTERIM)) cvEndpoint(false);
    else if (!heard && cvSilMs >= 6000) {
      cvEmptyStreak++;
      cvState = 'listening'; cvUtterMs = 0; cvSilMs = 0;
      cvMaybeDeaf();
      setStatus('● LISTENING — JUST SPEAK, SIR', 'live');
      syncSpeechUI();
    }
  }
}

function cvMaybeDeaf() {
  // Self-diagnosis: VAD heard real voice energy repeatedly but recognition
  // produced zero words. Either the mic stream is silent (wrong device) or the
  // browser engine is deaf — say so LOUDLY, and after 2 strikes switch to the
  // offline Whisper path for the rest of the session automatically.
  if (cvPeakHeard < cvSpeechThr()) {
    if (cvEmptyStreak >= 2) {
      pushMsg('err', 'My ears hear only silence, sir — the mic may be muted, unplugged, or the wrong device. Check Windows sound settings (right-click speaker → Sound settings → Input). I am still listening for HEY JARVIS.');
      setStatus('MIC SILENT — CHECK WINDOWS INPUT DEVICE', 'err');
    }
    return;
  }
  if (cvEmptyStreak >= 2 && SR && !cvForceWhisper) {
    cvForceWhisper = true;
    cvStopSR();
    pushMsg('jarvis', 'Browser ears are unreliable tonight, sir — switching to my offline ears. Keep talking, I am still listening.');
    setStatus('● LISTENING — JUST SPEAK, SIR (OFFLINE EARS)', 'live');
    syncSpeechUI();
  }
}

function cvEndpoint(fromMax) {
  const epoch = cvEpoch;
  if (SR && !cvForceWhisper) {
    // Include the latest interim so slow-final speech is never truncated.
    const text = (cvAcc + ' ' + cvInterim).trim();
    cvAcc = ''; cvInterim = ''; cvGotFinal = false; cvLastResultAt = 0;
    cvEpoch++;
    cvState = 'thinking';
    syncSpeechUI();
    if (!text) {
      // VAD heard energy but SR got no words — back to listening.
      cvState = 'listening';
      setStatus('● LISTENING — JUST SPEAK, SIR', 'live');
      syncSpeechUI();
      return;
    }
    cvEmptyStreak = 0; // a real utterance resets the deaf counter
    void epoch; void fromMax;
    handleVoiceText(text);
  }
  // Whisper path endpoint handled by recorder stop → upload → handleVoiceText.
}

function cvStartRecord() {
  if (cvMRActive || !mediaStream) return;
  try {
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : '';
    cvMR = mime ? new MediaRecorder(mediaStream, { mimeType: mime }) : new MediaRecorder(mediaStream);
  } catch (e) { return; }
  cvChunks = [];
  cvMR.ondataavailable = (e) => { if (e.data?.size) cvChunks.push(e.data); };
  cvMR.onstop = cvUpload;
  try { cvMR.start(250); cvMRActive = true; } catch (e) { cvMRActive = false; }
}
function cvStopRecord() {
  try { cvMR && cvMR.state !== 'inactive' && cvMR.stop(); } catch (e) { cvUpload(); }
}
async function cvUpload() {
  cvMRActive = false;
  const blob = new Blob(cvChunks, { type: 'audio/webm' });
  cvChunks = [];
  cvEpoch++;
  if (!converseOn) return;
  if (!blob.size) { converseResume(); return; }
  cvState = 'thinking';
  syncSpeechUI();
  setStatus('HEARD YOU — WORKING…', 'heard');
  try {
    const fd = new FormData();
    fd.append('audio', blob, 'utt.webm');
    const r = await fetch('/api/transcribe', { method: 'POST', body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || ('transcribe ' + r.status));
    const text = (data.text || '').trim();
    if (!text) throw new Error('empty transcript');
    cvEmptyStreak = 0;
    handleVoiceText(text);
  } catch (e) {
    setStatus('TRANSCRIBE HICCUP — SAY AGAIN, SIR', 'err');
    converseResume();
  }
}

async function converseStart(opts = {}) {
  if (converseOn && cvState !== 'off') return;
  const fromWake = !!opts.fromWake;
  try {
    autoVoiceOn();
    openChat();
    if (!fromWake) setStatus('TUNING MY EARS…', 'live');
    try { clearTimeout(wakeRestartTimer); } catch (e) {}
    stopRec(wakeRec); stopRec(cmdRec);
    if (sttMode === 'recording') stopWhisperRecord();
    sttMode = 'idle';
    await cvEnsureMic();
    converseOn = true;
    touchActivity();
    cvEpoch++; cvAcc = ''; cvInterim = ''; cvGotFinal = false; pendingVoice = null;
    cvPeakHeard = 0; cvEmptyStreak = 0;
    cvState = 'listening'; cvSilMs = 0; cvUtterMs = 0; cvBargeMs = 0;
    clearInterval(cvVadTimer);
    cvVadTimer = setInterval(cvTick, CV_TICK);
    cvStartSR();
    setStatus(SR ? '● LISTENING — JUST SPEAK, SIR' : '● LISTENING — JUST SPEAK, SIR (OFFLINE EARS)', 'live');
    syncSpeechUI();
    if (!fromWake) pushMsg('jarvis', 'Ears open, sir — just talk. I will wait till you finish, then answer and act. I will doze off after 30s of silence; call me any time.');
  } catch (e) {
    converseOn = false;
    wakeBlocked = true;
    setStatus('MIC BLOCKED — CLICK ONCE TO ENABLE EARS', 'err');
    pushMsg('err', 'Microphone blocked, sir — allow it in the address-bar lock icon, then click once anywhere and I will listen for HEY JARVIS.');
    syncSpeechUI();
    if (wakeOn) startWake();
  }
}
function sleepBackToWake(silent = false) {
  // Awake window over (30s silence) — park the full loop, resume cheap wake daemon.
  converseOn = false;
  cvState = 'off';
  clearInterval(cvVadTimer);
  cvStopSR();
  try { if (cvMRActive) { try { cvMR.stop(); } catch (e) {} cvMRActive = false; } } catch (e) {}
  try { window.jarvisStopSpeak?.(); } catch (e) {}
  pendingVoice = null;
  cvAcc = ''; cvInterim = ''; cvGotFinal = false;
  touchActivity();
  syncSpeechUI();
  if (!silent) setStatus("SLEEPING — SAY 'HEY JARVIS'", '');
  if (wakeOn) startWake();
}
function converseStop() {
  // Hard stop (errors only) — still falls back to wake daemon.
  sleepBackToWake();
}
function converseResume() {
  if (!converseOn) return;
  touchActivity();
  if (pendingVoice) {
    const t = pendingVoice; pendingVoice = null;
    handleVoiceText(t);
    return;
  }
  cvState = 'listening'; cvAcc = ''; cvInterim = ''; cvGotFinal = false;
  cvSilMs = 0; cvUtterMs = 0; cvBargeMs = 0; cvPeakHeard = 0;
  cvEpoch++;
  cvStartSR();
  setStatus(SR ? '● LISTENING — JUST SPEAK, SIR' : '● LISTENING — JUST SPEAK, SIR (OFFLINE EARS)', 'live');
  syncSpeechUI();
}
function converseAfterTurn() {
  // Agent done; spoken reply (started in handleFinal) is playing or about to.
  if (!converseOn || cvState === 'speaking') return; // idempotent: finally+handleFinal both call
  touchActivity();
  cvStopSR(); // no echo of his own voice
  cvState = 'speaking'; cvBargeMs = 0;
  setStatus('🔊 SPEAKING — TALK OVER ME TO INTERRUPT', 'heard');
  syncSpeechUI();
  clearTimeout(cvSpeakTimer);
  // If no TTS actually starts (nothing to say), don't hang in speaking.
  cvSpeakTimer = setTimeout(() => {
    if (converseOn && cvState === 'speaking') converseResume();
  }, 4000);
}
window.addEventListener('jarvis-speak-start', () => {
  if (!converseOn || cvState !== 'speaking') return;
  clearTimeout(cvSpeakTimer);
  // Genuine speech: allow full length, barge-in watches energy.
  cvSpeakTimer = setTimeout(() => {
    if (converseOn && cvState === 'speaking') converseResume();
  }, 45000);
});
window.addEventListener('jarvis-speak-end', () => {
  if (!converseOn || cvState !== 'speaking') return;
  clearTimeout(cvSpeakTimer);
  setTimeout(() => { if (converseOn && cvState === 'speaking') converseResume(); }, 450);
});

window.addEventListener('jarvis-speak-end', () => {
  if (!converseOn || cvState !== 'speaking') return;
  clearTimeout(cvSpeakTimer);
  touchActivity();
  setTimeout(() => { if (converseOn && cvState === 'speaking') converseResume(); }, 450);
});

micBtn?.addEventListener('click', () => {
  // Mic button retired — wake-word and typing are the two doors.
  // (Kept as interrupt only if the old button lingers in a cached page.)
  if (converseOn) {
    try { window.jarvisStopSpeak?.(); } catch (e) {}
    touchActivity();
    pendingVoice = null; cvEpoch++; cvAcc = ''; cvInterim = '';
    converseResume();
  }
});
// ◉ is a status indicator only — never starts/stops talk. A click only
// retries mic enable when blocked (autoplay / permission gate), nothing else.
wakeBtn?.addEventListener('click', () => {
  if (wakeBlocked || (!wakeStarted && !converseOn)) {
    wakeBlocked = false; sttFails = 0;
    wakeAutoStart(true);
  }
});
function wakeAutoStart(fromGesture = false) {
  // Browsers demand one gesture to open audio: try immediately, and if blocked
  // arm a one-time pointer/key listener that retries (that click enables ears only).
  if (converseOn) return;
  if (SR) {
    const ok = startWake();
    if (ok) return;
    if (!wakeBlocked && !wakeStarted) {
      // No SR engine (Firefox): ears stay offline, typing still works.
      return;
    }
  } else {
    setStatus('WAKE NEEDS CHROME/EDGE — TYPE BELOW, EARS OFFLINE', 'err');
    syncSpeechUI();
    return;
  }
  if (wakeBlocked || !wakeStarted) {
    if (fromGesture) return; // already tried in-gesture, wait for permission fix
    const retry = () => {
      try { document.removeEventListener('pointerdown', retry); } catch (e) {}
      try { document.removeEventListener('keydown', retry); } catch (e) {}
      wakeBlocked = false;
      startWake();
    };
    try { document.addEventListener('pointerdown', retry, { once: true }); } catch (e) {
      try { document.addEventListener('pointerdown', retry); } catch (e2) {}
    }
    try { document.addEventListener('keydown', retry, { once: true }); } catch (e) {}
    setStatus('CLICK ONCE TO ENABLE EARS — THEN SAY HEY JARVIS', 'live');
    syncSpeechUI();
  }
}
// Auto-sleep: 30s silence while awake → back to cheap wake-only.
setInterval(() => {
  if (!converseOn || cvState === 'off') return;
  if (cvState !== 'listening') { touchActivity(); return; }
  if (Date.now() - lastActivity >= WAKE_SLEEP_MS) {
    sleepBackToWake();
    pushMsg('jarvis', 'Doze mode, sir — say HEY JARVIS any time I am needed.');
  }
}, 1000);
window.addEventListener('jarvis-speak-end', () => {
  // Wake loop self-heals via onend-restart; just refresh the status line.
  if (wakeOn && !converseOn && sttMode === 'wake') setStatus("SLEEPING — SAY 'HEY JARVIS'", 'live');
  else touchActivity();
});
window.jarvisMic = {
  get mode() { return sttMode; },
  get wake() { return wakeOn; },
  get awake() { return converseOn; },
  get engine() { return SR ? 'browser' : 'whisper-record'; },
  startWake, stop: stopWakeToIdle, listenOnce,
  wakeUp: () => converseStart({ fromWake: true }),
  sleep: () => sleepBackToWake(),
};
// Boot: ears auto-arm, no button needed. First gesture unlocks audio if blocked.
syncSpeechUI();
setStatus('STARTING EARS — SAY HEY JARVIS…', 'live');
setTimeout(() => wakeAutoStart(false), 600);

/* ---------- boot ---------- */
refreshStatus();
setInterval(refreshStatus, 30000);
{
  const h = new Date().getHours();
  const part = h < 5 ? 'burning the midnight oil' : h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
  pushMsg('jarvis', `${part}, sir. Comm-link online and galaxy in view — ask me anything, have me brief you, or tell me something to remember and I will forge a new star for it.`);
}
