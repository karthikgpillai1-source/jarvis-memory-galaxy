/* JARVIS voice — speaks replies aloud + speech-to-speech support.
   Chain: neural backend voice (/api/speak, British male) -> browser speechSynthesis fallback.
   Off by default until first mic use (auto voice-on). Toggle persists.
   Exposes speaking state + stop + completion promise so the wake loop
   can pause while Jarvis talks and resume when he finishes (barge-in safe). */
const TTS_VOICE = 'en-GB-RyanNeural';
let voiceOn = localStorage.getItem('jarvis_voice') === 'on';
let currentAudio = null;
let currentUtterance = null;
let speaking = false;
let speakToken = 0;

function cleanForSpeech(s) {
  return (s || '').replace(/[*_`#>[\]()|]/g, '').replace(/\s+/g, ' ').trim().slice(0, 800);
}

function warmVoices() { try { speechSynthesis.getVoices(); } catch (e) {} }
if ('speechSynthesis' in window) {
  warmVoices();
  speechSynthesis.onvoiceschanged = warmVoices;
}

function setSpeaking(v) {
  speaking = v;
  try { window.dispatchEvent(new CustomEvent('jarvis-speaking', { detail: { speaking: v } })); } catch (e) {}
}

function browserSpeakPromise(text, token) {
  return new Promise((resolve) => {
    try {
      const synth = window.speechSynthesis;
      if (!synth) return resolve(false);
      synth.cancel();
      const u = new SpeechSynthesisUtterance(text);
      currentUtterance = u;
      const vs = synth.getVoices() || [];
      const pick =
        vs.find(v => /en-GB/i.test(v.lang) && /male|daniel|george|ryan|arthur/i.test(v.name)) ||
        vs.find(v => /Google UK English Male/i.test(v.name)) ||
        vs.find(v => /en-GB/i.test(v.lang)) ||
        vs.find(v => /David/i.test(v.name)) ||
        vs.find(v => /^en/i.test(v.lang));
      if (pick) u.voice = pick;
      u.rate = 0.95; u.pitch = 0.7; u.volume = 1;
      let done = false;
      const finish = (ok) => {
        if (done) return;
        done = true;
        if (currentUtterance === u) currentUtterance = null;
        resolve(token === speakToken && ok);
      };
      u.onend = () => finish(true);
      u.onerror = () => finish(false);
      // Safety: some browsers never fire onend for long text — cap at 60s.
      setTimeout(() => finish(false), 60000);
      synth.speak(u);
    } catch (e) { resolve(false); }
  });
}

async function neuralSpeak(text, token) {
  if (currentAudio) { try { currentAudio.pause(); } catch (e) {} currentAudio = null; }
  try {
    const r = await fetch('/api/speak?voice=' + encodeURIComponent(TTS_VOICE) + '&text=' + encodeURIComponent(text));
    if (token !== speakToken) return false; // barged-in while fetching
    if (!r.ok) throw new Error('tts ' + r.status);
    const blob = await r.blob();
    if (!blob.size) throw new Error('empty audio');
    if (token !== speakToken) return false;
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentAudio = audio;
    const done = await new Promise((resolve) => {
      let settled = false;
      const finish = (ok) => {
        if (settled) return;
        settled = true;
        try { URL.revokeObjectURL(url); } catch (e) {}
        if (currentAudio === audio) currentAudio = null;
        resolve(token === speakToken && ok);
      };
      audio.onended = () => finish(true);
      audio.onerror = () => finish(false);
      audio.play().catch(() => finish(false));
      setTimeout(() => finish(false), 90000); // safety cap
    });
    return done;
  } catch (e) { return false; }
}

window.jarvisSpeak = async function (raw) {
  if (!voiceOn) return false;
  const text = cleanForSpeech(raw);
  if (!text) return false;
  const token = ++speakToken;
  // Barge-in: kill anything currently playing before starting.
  try { speechSynthesis?.cancel(); } catch (e) {}
  if (currentAudio) { try { currentAudio.pause(); } catch (e) {} currentAudio = null; }
  currentUtterance = null;
  setSpeaking(true);
  try { window.dispatchEvent(new CustomEvent('jarvis-speak-start')); } catch (e) {}
  let ok = await neuralSpeak(text, token);
  if (!ok && token === speakToken) ok = await browserSpeakPromise(text, token);
  if (token === speakToken) {
    setSpeaking(false);
    try { window.dispatchEvent(new CustomEvent('jarvis-speak-end', { detail: { ok } })); } catch (e) {}
  }
  return ok;
};

window.jarvisStopSpeak = function () {
  speakToken++; // invalidate in-flight speak promises
  try { speechSynthesis?.cancel(); } catch (e) {}
  if (currentAudio) { try { currentAudio.pause(); } catch (e) {} currentAudio = null; }
  currentUtterance = null;
  if (speaking) {
    setSpeaking(false);
    try { window.dispatchEvent(new CustomEvent('jarvis-speak-end', { detail: { ok: false, interrupted: true } })); } catch (e) {}
  }
};

function syncVoiceBtn() {
  const b = document.getElementById('btn-voice');
  if (b) { b.classList.toggle('on', voiceOn); b.textContent = voiceOn ? '🔊' : '🔇'; }
}
window.jarvisVoice = {
  get enabled() { return voiceOn; },
  get speaking() { return speaking; },
  isSpeaking() { return speaking; },
  enable() {
    if (!voiceOn) {
      voiceOn = true;
      localStorage.setItem('jarvis_voice', 'on');
      syncVoiceBtn();
    }
    return voiceOn;
  },
  stop() { window.jarvisStopSpeak(); },
  toggle() {
    voiceOn = !voiceOn;
    localStorage.setItem('jarvis_voice', voiceOn ? 'on' : 'off');
    if (!voiceOn) window.jarvisStopSpeak();
    syncVoiceBtn();
    return voiceOn;
  },
};
document.getElementById('btn-voice')?.addEventListener('click', () => {
  const on = window.jarvisVoice.toggle();
  if (on) window.jarvisSpeak('Voice systems online, sir. At your service.');
});
syncVoiceBtn();
