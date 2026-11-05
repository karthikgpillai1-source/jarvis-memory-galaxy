import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/* ================= CONFIG ================= */
const SPHERE_RADIUS = 5;
const SECTOR = {
  personal:  { color: 0x00d4ff, label: 'PERSONAL', css: '#00d4ff' },
  work:      { color: 0x00ff9d, label: 'WORK',     css: '#00ff9d' },
  ideas:     { color: 0xffb300, label: 'IDEAS',    css: '#ffb300' },
  important: { color: 0xff3b5c, label: 'CRITICAL', css: '#ff3b5c' },
};
const $ = id => document.getElementById(id);

/* ================= DOM ================= */
const container = $('canvas-container'), labelLayer = $('label-layer');
const bootEl = $('boot'), bootFill = $('boot-fill'), bootLog = $('boot-log');
const countEl = $('memory-count'), timeEl = $('current-time'), fpsEl = $('fps');
const pwrFill = $('pwr-fill'), pwrVal = $('pwr-val');
const insp = $('inspector'), dSector = $('detail-sector'), dTitle = $('detail-title'),
      dId = $('detail-id'), dDate = $('detail-date'), dCoords = $('detail-coords'), dContent = $('detail-content'),
      relatedEl = $('related');
const hoverTag = $('hover-tag'), hoverText = $('hover-text'), hoverDot = $('hover-dot');
const logBox = $('jarvis-log'), toastBox = $('toast-container');
const cmdInput = $('cmd-input'), searchInput = $('search-input'), searchBox = $('search-results');

/* ================= STATE ================= */
let memories = [], nodes = [], hitProxies = [];
let activeSector = 'all', selectedId = null, hoveredId = null;
let autoRotate = true, showWire = true, showAura = true, showLinks = true;
let bloomOn = true, showLabels = false, soundOn = true, touring = false, tourTimer = null;
let spinSpeed = 0.28, linkDist = 3.4;
let addSector = 'personal';
let introDone = false;
const bootTime = Date.now();

/* Titles of the retired factory demo data (v1). Anything matching these was
   never created by the user, so it is purged once on load. Real user memories
   are always preserved. */
const LEGACY_TITLES = new Set([
  "First Boot Sequence", "Arc Reactor Design v3", "Malibu House Party",
  "Time Travel Theory", "Vibranium Analysis", "Friday Movie Night",
  "Nano-tech Suit", "Ultron Protocol", "Stark Expo Keynote",
  "Shawarma After Battle", "Hulkbuster Upgrade", "House Party Protocol",
  "New Element Synthesis", "Pepper's Birthday", "EDITH Glasses",
]);
let legacyPurged = 0;

function loadMemories() {
  try {
    const raw = localStorage.getItem('jarvis_memories_v2');
    if (raw) {
      memories = JSON.parse(raw);
      const before = memories.length;
      memories = memories.filter(m => !LEGACY_TITLES.has(m.title));
      legacyPurged = before - memories.length;
      if (legacyPurged > 0) saveMemories();
      return;
    }
  } catch(e) {}
  memories = []; // fresh galaxy starts empty — JARVIS helps you seed it
  saveMemories();
}
function saveMemories() { try { localStorage.setItem('jarvis_memories_v2', JSON.stringify(memories)); } catch(e) {} }

/* ================= AUDIO ================= */
let actx = null;
function blip(f = 660, d = 0.18, type = 'sine', vol = 0.07) {
  if (!soundOn) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === 'suspended') actx.resume();
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = type; o.frequency.value = f;
    g.gain.setValueAtTime(vol, actx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime + d);
    o.connect(g); g.connect(actx.destination);
    o.start(); o.stop(actx.currentTime + d + 0.02);
  } catch(e) {}
}
const sndSelect = () => { blip(880, .16); setTimeout(() => blip(1320, .12, 'sine', .05), 70); };
const sndStore = () => { blip(520, .12, 'triangle'); setTimeout(() => blip(780, .12, 'triangle'), 90); setTimeout(() => blip(1040, .2, 'sine', .06), 180); };

/* ================= LOG + TOAST ================= */
function jlog(msg, cls = '') {
  const d = document.createElement('div');
  d.className = 'jlog ' + cls;
  d.textContent = `▸ ${msg}`;
  logBox.appendChild(d);
  while (logBox.children.length > 5) logBox.firstChild.remove();
  setTimeout(() => { d.style.opacity = '0'; d.style.transition = 'opacity .5s'; }, 5200);
  setTimeout(() => d.remove(), 5800);
}
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = 'toast ' + type; el.textContent = msg;
  toastBox.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .4s'; }, 2600);
  setTimeout(() => el.remove(), 3100);
}

/* ================= THREE CORE ================= */
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020409);
scene.fog = new THREE.FogExp2(0x020409, 0.012);

const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 9, 30);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.minDistance = 7; controls.maxDistance = 34; controls.enablePan = false;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 1.1, 0.75, 0.12);
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

scene.add(new THREE.AmbientLight(0x3a5a78, 1.4));
const coreLight = new THREE.PointLight(0x00d4ff, 120, 40, 1.6);
scene.add(coreLight);
const rim = new THREE.DirectionalLight(0x3388ff, 2.0); rim.position.set(12, 9, 6); scene.add(rim);
const rim2 = new THREE.DirectionalLight(0xff3b5c, 0.35); rim2.position.set(-10, -6, -8); scene.add(rim2);

const galaxy = new THREE.Group(); scene.add(galaxy);

/* ---------- textures ---------- */
function radialTex(inner, outer) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const x = c.getContext('2d');
  const g = x.createRadialGradient(128,128,0,128,128,128);
  g.addColorStop(0, inner); g.addColorStop(0.35, outer); g.addColorStop(1, 'rgba(0,0,0,0)');
  x.fillStyle = g; x.fillRect(0,0,256,256);
  return new THREE.CanvasTexture(c);
}
function tickRingTex() {
  const c = document.createElement('canvas'); c.width = 1024; c.height = 128;
  const x = c.getContext('2d'); x.clearRect(0,0,1024,128);
  for (let i = 0; i < 120; i++) {
    const big = i % 10 === 0;
    x.fillStyle = big ? 'rgba(0,229,255,1)' : 'rgba(0,229,255,.45)';
    x.fillRect(i * (1024/120), big ? 10 : 34, big ? 4 : 2, big ? 108 : 60);
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = THREE.RepeatWrapping; return t;
}
const glowCyan = radialTex('#ffffff', 'rgba(0,229,255,.55)');
const glowCache = {};
function glowFor(css) { return glowCache[css] ??= radialTex('#ffffff', css); }
const nebTexA = radialTex('rgba(40,80,255,.5)', 'rgba(0,229,255,.18)');
const nebTexB = radialTex('rgba(150,60,255,.4)', 'rgba(255,59,150,.12)');

/* ---------- nebula backdrop ---------- */
[[ -34, 10, -70, 60, nebTexA, .5 ], [ 38, -14, -66, 70, nebTexB, .42 ],
 [ 0, 26, -80, 80, nebTexA, .35 ], [ -20, -24, 60, 55, nebTexB, .3 ]].forEach(([x,y,z,s,tex,op]) => {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: op, blending: THREE.AdditiveBlending, depthWrite: false }));
  sp.position.set(x,y,z); sp.scale.set(s,s,1); scene.add(sp);
});

/* ---------- starfield (colored) ---------- */
{
  const N = 2400, pos = new Float32Array(N*3), col = new Float32Array(N*3);
  const palette = [new THREE.Color(0xffffff), new THREE.Color(0x9fdcff), new THREE.Color(0x00e5ff), new THREE.Color(0xffc37a)];
  for (let i = 0; i < N; i++) {
    const r = 70 + Math.random()*110, th = Math.random()*Math.PI*2, ph = Math.acos(2*Math.random()-1);
    pos.set([r*Math.sin(ph)*Math.cos(th), r*Math.sin(ph)*Math.sin(th), r*Math.cos(ph)], i*3);
    const c = palette[(Math.random()*palette.length)|0];
    col.set([c.r, c.g, c.b], i*3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  g.setAttribute('color', new THREE.BufferAttribute(col,3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 0.5, vertexColors: true, transparent: true, opacity: 0.85, sizeAttenuation: true })));
}

/* ---------- arc-reactor core (shader) ---------- */
const coreUniforms = { uTime: { value: 0 }, uColor: { value: new THREE.Color(0x00e5ff) } };
const coreMesh = new THREE.Mesh(
  new THREE.SphereGeometry(1.12, 48, 48),
  new THREE.ShaderMaterial({
    uniforms: coreUniforms,
    vertexShader: `varying vec3 vN; varying vec3 vP; void main(){ vN=normalize(normalMatrix*normal); vP=position; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform float uTime; uniform vec3 uColor; varying vec3 vN; varying vec3 vP;
      void main(){
        float bands = sin(vP.y*9.0 - uTime*3.0)*.5+.5;
        float pulse = .75 + .25*sin(uTime*2.4);
        vec3 v = normalize(vec3(0.,0.,1.));
        float fr = pow(1.-abs(dot(vN,v)), 1.6);
        vec3 col = uColor*(0.55+bands*0.5)*pulse + vec3(1.)*fr*0.9;
        gl_FragColor = vec4(col, 1.0);
      }`
  })
);
galaxy.add(coreMesh);
const coreCage = new THREE.Mesh(new THREE.IcosahedronGeometry(1.5, 1),
  new THREE.MeshBasicMaterial({ color: 0x00e5ff, wireframe: true, transparent: true, opacity: 0.55 }));
galaxy.add(coreCage);
const coreCage2 = new THREE.Mesh(new THREE.IcosahedronGeometry(1.85, 1),
  new THREE.MeshBasicMaterial({ color: 0x0090b3, wireframe: true, transparent: true, opacity: 0.22 }));
galaxy.add(coreCage2);
const coreGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowCyan, color: 0x00d4ff, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
coreGlow.scale.set(7,7,1); galaxy.add(coreGlow);

/* core orbit sparks */
let sparks, sparkVel;
{
  const N = 260, pos = new Float32Array(N*3); sparkVel = new Float32Array(N);
  for (let i=0;i<N;i++){ const a=Math.random()*Math.PI*2, r=1.7+Math.random()*1.4;
    pos.set([Math.cos(a)*r,(Math.random()-.5)*1.6,Math.sin(a)*r], i*3); sparkVel[i]=.4+Math.random()*1.2; }
  const g=new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  sparks=new THREE.Points(g,new THREE.PointsMaterial({color:0x8ff5ff,size:.07,transparent:true,opacity:.9,blending:THREE.AdditiveBlending,depthWrite:false}));
  galaxy.add(sparks);
}

/* ---------- lattice + shell ---------- */
const lattice = new THREE.Group(); galaxy.add(lattice);
const lat1 = new THREE.LineSegments(
  new THREE.WireframeGeometry(new THREE.SphereGeometry(SPHERE_RADIUS, 36, 24)),
  new THREE.LineBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.10 }));
const lat2 = new THREE.LineSegments(
  new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(SPHERE_RADIUS, 2)),
  new THREE.LineBasicMaterial({ color: 0x66f0ff, transparent: true, opacity: 0.06 }));
lattice.add(lat1, lat2);

const shellUniforms = { uTime: { value: 0 } };
const shell = new THREE.Mesh(new THREE.SphereGeometry(SPHERE_RADIUS*1.015, 64, 64),
  new THREE.ShaderMaterial({ uniforms: shellUniforms, transparent: true, side: THREE.BackSide,
    blending: THREE.AdditiveBlending, depthWrite: false,
    vertexShader: `varying vec3 vN; varying vec3 vW; void main(){ vN=normalize(mat3(modelMatrix)*normal); vec4 w=modelMatrix*vec4(position,1.); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }`,
    fragmentShader: `uniform float uTime; varying vec3 vN; varying vec3 vW;
      void main(){ vec3 V=normalize(cameraPosition-vW); float f=pow(1.-abs(dot(normalize(vN),V)),2.6);
        float scan=.85+.15*sin(vW.y*6.-uTime*1.4); vec3 col=vec3(.05,.75,1.)*f*1.6*scan; gl_FragColor=vec4(col,f*.9); }` }));
galaxy.add(shell);

/* ---------- HUD rings ---------- */
const hudRings = [];
function hudRing(radius, texRepeat, opacity, tilt, speed) {
  const tex = tickRingTex(); tex.repeat.set(texRepeat,1);
  const m = new THREE.Mesh(new THREE.RingGeometry(radius-0.09, radius+0.09, 128, 1),
    new THREE.MeshBasicMaterial({ map: tex, color: 0x00e5ff, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
  m.rotation.x = tilt; galaxy.add(m);
  hudRings.push({ mesh: m, speed, tex });
  const thin = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.012, 8, 160),
    new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: opacity*0.9 }));
  thin.rotation.x = tilt; galaxy.add(thin); hudRings.push({ mesh: thin, speed: speed*0.6 });
}
hudRing(6.35, 6, 0.85, Math.PI/2.12, 0.12);
hudRing(7.05, 4, 0.45, Math.PI/1.88, -0.09);
hudRing(7.85, 3, 0.28, Math.PI/2.35, 0.06);

/* ---------- dust ---------- */
let dust, dustBase;
{
  const N = 750, pos = new Float32Array(N*3); dustBase = new Float32Array(N*3);
  for (let i=0;i<N;i++){ const r=SPHERE_RADIUS+0.6+Math.random()*6, th=Math.random()*Math.PI*2, ph=Math.acos(2*Math.random()-1);
    const p=[r*Math.sin(ph)*Math.cos(th),r*Math.sin(ph)*Math.sin(th),r*Math.cos(ph)];
    pos.set(p,i*3); dustBase.set(p,i*3); }
  const g=new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  dust=new THREE.Points(g,new THREE.PointsMaterial({color:0x35d5ff,size:.055,transparent:true,opacity:.5,blending:THREE.AdditiveBlending,depthWrite:false}));
  galaxy.add(dust);
}

/* ---------- links / beam / halo / shockwaves ---------- */
let linkLines = null, selLines = null;
function refreshLinks() {
  for (const l of [linkLines, selLines]) if (l) { galaxy.remove(l); l.geometry.dispose(); l.material.dispose(); }
  linkLines = selLines = null;
  const vis = nodes.filter(n => n.visible);
  if (showLinks && vis.length) {
    const pts = [];
    for (let i=0;i<vis.length;i++) for (let j=i+1;j<vis.length;j++)
      if (vis[i].position.distanceTo(vis[j].position) < linkDist) pts.push(...vis[i].position.toArray(), ...vis[j].position.toArray());
    if (pts.length) {
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts),3));
      linkLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x00c8f0, transparent: true, opacity: 0.2, blending: THREE.AdditiveBlending, depthWrite: false }));
      galaxy.add(linkLines);
    }
  }
  if (selectedId) {
    const s = nodes.find(n => n.userData.mid === selectedId);
    if (s) {
      const pts = [];
      for (const n of vis) { if (n === s) continue;
        if (n.position.distanceTo(s.position) < linkDist*1.5) pts.push(...s.position.toArray(), ...n.position.toArray()); }
      if (pts.length) {
        const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts),3));
        selLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
        galaxy.add(selLines);
      }
    }
  }
}
const beamGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0,5,0)]);
const beam = new THREE.Line(beamGeo, new THREE.LineBasicMaterial({ color: 0x9ff3ff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
galaxy.add(beam);
const halo = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.022, 10, 56),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }));
halo.visible = false; scene.add(halo);
const halo2 = new THREE.Mesh(new THREE.TorusGeometry(0.58, 0.012, 8, 56),
  new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.7 }));
halo2.visible = false; scene.add(halo2);

const shocks = [];
function shockwave(worldPos, color = 0x00e5ff, max = 2.4) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.36, 48),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
  m.position.copy(worldPos); scene.add(m); shocks.push({ m, t: 0, max });
}
/* burst particles */
const bursts = [];
function burst(worldPos, color = 0x00e5ff, n = 90) {
  const pos = new Float32Array(n*3), vel = new Float32Array(n*3);
  for (let i=0;i<n;i++){ pos.set([worldPos.x,worldPos.y,worldPos.z],i*3);
    const v = new THREE.Vector3().randomDirection().multiplyScalar(2+Math.random()*4); vel.set([v.x,v.y,v.z],i*3); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  const p = new THREE.Points(g, new THREE.PointsMaterial({ color, size: 0.12, transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false }));
  scene.add(p); bursts.push({ p, vel, life: 1 });
}

/* ================= MEMORY NODES ================= */
const nodeGeo = new THREE.OctahedronGeometry(0.17);
const hitGeo = new THREE.SphereGeometry(0.5, 8, 8);
const hitMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });

function fibPoint(i, total, r) {
  if (total <= 1) return new THREE.Vector3(0, r, 0);
  const ga = Math.PI * (3 - Math.sqrt(5));
  const y = 1 - (i/(total-1))*2, rr = Math.sqrt(Math.max(0,1-y*y)), th = ga*i;
  return new THREE.Vector3(Math.cos(th)*rr*r, y*r, Math.sin(th)*rr*r);
}
function randSphere(r) {
  const v = new THREE.Vector3().randomDirection().multiplyScalar(r); return v;
}
function makeNode(mem, position, animate = false) {
  const s = SECTOR[mem.sector] || SECTOR.personal;
  const mat = new THREE.MeshBasicMaterial({ color: s.color });
  const mesh = new THREE.Mesh(nodeGeo, mat);
  mesh.position.copy(position);
  const big = mem.sector === 'important' ? 1.4 : 1;
  mesh.scale.setScalar(animate ? 0.01 : big);
  mesh.userData = { mid: mem.id, phase: Math.random()*Math.PI*2, big, target: big };
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowFor(s.css), color: s.color, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false }));
  spr.scale.set(1.25*big, 1.25*big, 1); mesh.add(spr);
  mesh.userData.spr = spr;
  const hit = new THREE.Mesh(hitGeo, hitMat); hit.userData.mid = mem.id; mesh.add(hit);
  galaxy.add(mesh); nodes.push(mesh); hitProxies.push(hit);
  return mesh;
}
function rebuildNodes() {
  for (const n of nodes) { galaxy.remove(n); n.material.dispose(); n.userData.spr.material.dispose(); }
  nodes = []; hitProxies = [];
  memories.forEach((m, i) => makeNode(m, fibPoint(i, memories.length, SPHERE_RADIUS).setLength(SPHERE_RADIUS)));
  applyFilter();
}
function nearest(mem, k = 5) {
  const s = nodes.find(n => n.userData.mid === mem.id); if (!s) return [];
  return nodes.filter(n => n.visible && n.userData.mid !== mem.id)
    .map(n => ({ n, d: n.position.distanceTo(s.position), mem: memories.find(m => m.id === n.userData.mid) }))
    .filter(x => x.mem).sort((a,b) => a.d-b.d).slice(0,k);
}

/* ================= FILTER / SEARCH ================= */
function applyFilter() {
  const q = searchInput.value.trim().toLowerCase();
  for (const n of nodes) {
    const m = memories.find(x => x.id === n.userData.mid);
    if (!m) { n.visible = false; continue; }
    n.visible = (activeSector === 'all' || m.sector === activeSector) &&
      (!q || m.title.toLowerCase().includes(q) || m.content.toLowerCase().includes(q));
    n.userData.spr.visible = showAura && n.visible;
  }
  lattice.visible = showWire;
  if (dust) dust.visible = showAura;
  refreshLinks(); updateCounts(); renderSearch();
}
function updateCounts() {
  const vis = nodes.filter(n => n.visible).length;
  countEl.textContent = `${vis}`;
  const emptyHint = document.getElementById('empty-hint');
  if (emptyHint) emptyHint.style.display = memories.length ? 'none' : 'block';
  const by = s => memories.filter(m => m.sector === s).length;
  $('c-all').textContent = memories.length;
  $('c-personal').textContent = by('personal');
  $('c-work').textContent = by('work');
  $('c-ideas').textContent = by('ideas');
  $('c-important').textContent = by('important');
}
function renderSearch() {
  const q = searchInput.value.trim().toLowerCase();
  searchBox.innerHTML = '';
  if (!q) return;
  memories.filter(m => m.title.toLowerCase().includes(q) || m.content.toLowerCase().includes(q)).slice(0,9)
    .forEach(h => {
      const s = SECTOR[h.sector] || SECTOR.personal;
      const d = document.createElement('div'); d.className = 'sr-item';
      d.innerHTML = `<div class="sr-t">${h.title}</div><div class="sr-p">${h.content.slice(0,70)}…</div><div class="sr-s" style="color:${s.css}">◆ ${s.label}</div>`;
      d.onclick = () => { selectMemory(h.id, true); sndSelect(); };
      searchBox.appendChild(d);
    });
}

/* ================= SELECT ================= */
function selectMemory(id, focus = false) {
  selectedId = id;
  const m = memories.find(x => x.id === id); if (!m) return;
  const s = SECTOR[m.sector] || SECTOR.personal;
  const node = nodes.find(n => n.userData.mid === id);
  dSector.textContent = '◆ ' + s.label; dSector.style.color = s.css; dSector.style.borderColor = s.css + '66';
  dTitle.textContent = m.title;
  dId.textContent = 'ID ' + id.slice(-8).toUpperCase();
  const dt = new Date(m.date);
  dDate.textContent = dt.toLocaleDateString() + ' ' + dt.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  dContent.textContent = m.content;
  if (node) {
    const p = node.position;
    dCoords.textContent = `X ${p.x.toFixed(2)} · Y ${p.y.toFixed(2)} · Z ${p.z.toFixed(2)} · R ${p.length().toFixed(2)}`;
    const wp = new THREE.Vector3(); node.getWorldPosition(wp); shockwave(wp, s.color, 2.0);
  }
  relatedEl.innerHTML = '';
  nearest(m).forEach(({ mem: r }) => {
    const rs = SECTOR[r.sector] || SECTOR.personal;
    const d = document.createElement('div'); d.className = 'rel';
    d.innerHTML = `<span class="dot" style="background:${rs.css};box-shadow:0 0 8px ${rs.css}"></span><span>${r.title}</span>`;
    d.onclick = () => selectMemory(r.id, true);
    relatedEl.appendChild(d);
  });
  if (!nearest(m).length) relatedEl.innerHTML = '<div class="rel" style="opacity:.5">No close links in range</div>';
  insp.classList.add('visible', 'has-target');
  halo.visible = halo2.visible = true;
  refreshLinks();
  jlog(`TARGET LOCKED — "${m.title}" [${s.label}]`);
  if (focus) focusOn(id);
}
function deselect() {
  selectedId = null; halo.visible = halo2.visible = false;
  beam.material.opacity = 0;
  insp.classList.remove('visible', 'has-target');
  dTitle.textContent = 'NO TARGET LOCKED'; dContent.textContent = 'Click any star in the sphere to inspect its memory. Use scroll to dive closer.';
  refreshLinks();
}
function focusOn(id) {
  const n = nodes.find(x => x.userData.mid === id); if (!n) return;
  const wp = new THREE.Vector3(); n.getWorldPosition(wp);
  const t0 = performance.now(), fromT = controls.target.clone(), toT = wp.clone().multiplyScalar(0.3);
  const fromC = camera.position.clone();
  const dir = camera.position.clone().sub(controls.target).normalize();
  const toC = toT.clone().add(dir.multiplyScalar(9.5));
  (function step(t) {
    const k = Math.min(1, (t-t0)/700), e = 1-Math.pow(1-k,3);
    controls.target.lerpVectors(fromT, toT, e); camera.position.lerpVectors(fromC, toC, e);
    if (k < 1) requestAnimationFrame(step);
  })(t0);
}

/* ================= INTERACTION ================= */
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
let downX = 0, downY = 0;
function castAt(e) {
  ptr.x = (e.clientX/innerWidth)*2-1; ptr.y = -(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(ptr, camera);
  const hits = ray.intersectObjects(hitProxies.filter(h => {
    const n = nodes.find(x => x.userData.mid === h.userData.mid); return n && n.visible;
  }), false);
  return hits.length ? hits[0].object.userData.mid : null;
}
renderer.domElement.addEventListener('pointermove', e => {
  const id = castAt(e);
  hoveredId = id;
  if (id) {
    const m = memories.find(x => x.id === id), s = SECTOR[m?.sector] || SECTOR.personal;
    document.body.style.cursor = 'pointer';
    hoverTag.style.display = 'flex';
    hoverTag.style.left = (e.clientX+16)+'px'; hoverTag.style.top = (e.clientY+12)+'px';
    hoverText.textContent = m ? m.title.toUpperCase().slice(0,34) : '';
    hoverDot.style.background = s.css; hoverDot.style.boxShadow = `0 0 8px ${s.css}`;
    hoverTag.style.borderColor = s.css;
  } else { document.body.style.cursor = 'grab'; hoverTag.style.display = 'none'; }
});
renderer.domElement.addEventListener('pointerdown', e => { downX = e.clientX; downY = e.clientY; stopTour(); });
renderer.domElement.addEventListener('pointerup', e => {
  if (Math.hypot(e.clientX-downX, e.clientY-downY) > 6) return;
  const id = castAt(e);
  if (id) { selectMemory(id); sndSelect(); } else deselect();
});
renderer.domElement.addEventListener('wheel', stopTour, { passive: true });

/* ================= UI WIRING ================= */
document.querySelectorAll('.dock-btn').forEach(b => b.onclick = () => {
  if (b.dataset.tab === '__chat') return; // handled by chat.js comm-link
  document.querySelectorAll('.dock-btn').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); $('tab-'+b.dataset.tab).classList.add('active'); blip(600,.1);
});
function holoToggle(id, val, fn) {
  const b = $(id);
  b.classList.toggle('on', val);
  b.onclick = () => { fn(); b.classList.toggle('on'); blip(520,.1); };
}
holoToggle('btn-rotate-auto', autoRotate, () => autoRotate = !autoRotate);
holoToggle('btn-wireframe', showWire, () => { showWire = !showWire; applyFilter(); });
holoToggle('btn-particles', showAura, () => { showAura = !showAura; applyFilter(); });
holoToggle('btn-connections', showLinks, () => { showLinks = !showLinks; applyFilter(); });
$('btn-bloom').classList.add('on');
$('btn-bloom').onclick = () => { bloomOn = !bloomOn; bloomPass.enabled = bloomOn; $('btn-bloom').classList.toggle('on', bloomOn); blip(520,.1); };
$('btn-labels').onclick = () => { showLabels = !showLabels; $('btn-labels').classList.toggle('on', showLabels); };
$('spin-speed').oninput = e => { spinSpeed = e.target.value/100; $('spin-val').textContent = spinSpeed.toFixed(2); };
$('bloom-strength').oninput = e => { bloomPass.strength = e.target.value/100; $('bloom-val').textContent = bloomPass.strength.toFixed(2); };
$('zoom-range').oninput = e => {
  const d = e.target.value/10;
  const dir = camera.position.clone().sub(controls.target).normalize();
  camera.position.copy(controls.target).add(dir.multiplyScalar(d));
};
let linkT = null;
$('link-range').oninput = e => { linkDist = e.target.value/10; $('link-val').textContent = linkDist.toFixed(1);
  clearTimeout(linkT); linkT = setTimeout(refreshLinks, 160); };

document.querySelectorAll('.sector-card').forEach(b => b.onclick = () => {
  document.querySelectorAll('.sector-card').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); activeSector = b.dataset.sector; applyFilter(); blip(700,.1);
  jlog(`SECTOR FILTER → ${activeSector.toUpperCase()}`);
});
document.querySelectorAll('#sector-seg button').forEach(b => b.onclick = e => {
  e.preventDefault();
  document.querySelectorAll('#sector-seg button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); addSector = b.dataset.v; blip(700,.08);
});
function revealMemory(id) {
  // A star hidden by the sector filter looks like a failed store/focus —
  // drop back to ALL so the user actually sees what changed.
  const m = memories.find(x => x.id === id);
  if (m && activeSector !== 'all' && m.sector !== activeSector) setSectorFilter('all');
}
function commitMemoryObject(mem, announce = true) {
  if (!mem || !mem.title || !mem.content) return null;
  if (!mem.id || memories.some(m => m.id === mem.id)) mem.id = 'mem-' + Date.now();
  if (!SECTOR[mem.sector]) mem.sector = 'personal';
  mem.title = String(mem.title).slice(0, 80);
  mem.content = String(mem.content).slice(0, 2000);
  mem.date = mem.date || new Date().toISOString();
  memories.push(mem); saveMemories();
  revealMemory(mem.id);
  const n = makeNode(mem, randSphere(SPHERE_RADIUS), true);
  const wp = new THREE.Vector3(); n.getWorldPosition(wp);
  burst(wp, SECTOR[mem.sector].color); shockwave(wp, SECTOR[mem.sector].color);
  applyFilter(); selectMemory(mem.id, true);
  if (announce) { sndStore(); toast('MEMORY COMMITTED TO CORE', 'success'); jlog(`STORED "${mem.title}" → ${mem.sector.toUpperCase()} SECTOR`); }
  return mem;
}
$('memory-form').addEventListener('submit', e => {
  e.preventDefault();
  const title = $('memory-title').value.trim(), content = $('memory-content').value.trim();
  if (!title || !content) return;
  commitMemoryObject({ id: 'mem-'+Date.now(), title, content, sector: addSector, date: new Date().toISOString() });
  e.target.reset();
});
searchInput.addEventListener('input', applyFilter);
$('close-detail').onclick = deselect;
$('btn-focus').onclick = () => selectedId && (focusOn(selectedId), blip(760,.1));
function removeMemoryObject(id, announce = true) {
  const m = memories.find(x => x.id === id);
  if (!m) return null;
  const n = nodes.find(x => x.userData.mid === id);
  if (n) { const wp = new THREE.Vector3(); n.getWorldPosition(wp); burst(wp, 0xff3b5c, 120); }
  memories = memories.filter(x => x.id !== id);
  const i = nodes.findIndex(x => x.userData.mid === id);
  if (i >= 0) { galaxy.remove(nodes[i]); nodes.splice(i, 1); }
  hitProxies = hitProxies.filter(h => h.userData.mid !== id);
  saveMemories();
  if (selectedId === id) deselect(); else applyFilter();
  if (announce) { toast('MEMORY PURGED', 'error'); jlog(`PURGED "${m.title}"`, 'err'); blip(220, .25, 'sawtooth', .06); }
  return m;
}
function editMemoryObject(id, patch = {}, announce = true) {
  const m = memories.find(x => x.id === id);
  if (!m) return null;
  if (patch.title?.trim()) m.title = patch.title.trim().slice(0, 80);
  if (patch.content?.trim()) m.content = patch.content.trim().slice(0, 2000);
  if (patch.sector && SECTOR[patch.sector]) {
    m.sector = patch.sector;
    const n = nodes.find(x => x.userData.mid === id);
    if (n) {
      const s = SECTOR[m.sector];
      n.material.color.setHex(s.color);
      n.userData.spr.material.color.setHex(s.color);
      n.userData.spr.material.map = glowFor(s.css);
    }
  }
  saveMemories(); revealMemory(id); applyFilter();
  if (selectedId === id) selectMemory(id);
  if (announce) { toast('MEMORY UPDATED', 'success'); jlog(`UPDATED "${m.title}"`); blip(880, .12); }
  return m;
}
function setSectorFilter(sector) {
  if (!['all', 'personal', 'work', 'ideas', 'important'].includes(sector)) return false;
  document.querySelector(`.sector-card[data-sector="${sector}"]`)?.click();
  return true;
}
function galaxyCommand(cmd) {
  const on = (id, v) => $(id)?.classList.toggle('on', v);
  switch (cmd) {
    case 'spin_on': autoRotate = true; on('btn-rotate-auto', true); jlog('AUTO-SPIN ENGAGED'); break;
    case 'spin_off': autoRotate = false; on('btn-rotate-auto', false); jlog('AUTO-SPIN OFFLINE'); break;
    case 'bloom_on': bloomOn = true; bloomPass.enabled = true; on('btn-bloom', true); break;
    case 'bloom_off': bloomOn = false; bloomPass.enabled = false; on('btn-bloom', false); break;
    case 'labels_on': showLabels = true; on('btn-labels', true); break;
    case 'labels_off': showLabels = false; on('btn-labels', false); break;
    case 'tour_start': if (!touring) $('btn-tour').click(); break;
    case 'tour_stop': if (touring) stopTour(); break;
    case 'reshuffle': $('btn-reshuffle').click(); break;
    case 'cinematic_on': document.body.classList.add('cine'); break;
    case 'cinematic_off': document.body.classList.remove('cine'); break;
    default: return false;
  }
  blip(520, .1);
  return true;
}
$('btn-delete').onclick = () => { if (selectedId) removeMemoryObject(selectedId); };
$('btn-edit').onclick = () => {
  if (!selectedId) return;
  const m = memories.find(x => x.id === selectedId); if (!m) return;
  const nt = prompt('EDIT TITLE:', m.title); if (nt === null) return;
  const nc = prompt('EDIT CONTENT:', m.content); if (nc === null) return;
  m.title = nt.trim() || m.title; m.content = nc.trim() || m.content;
  saveMemories(); selectMemory(m.id); applyFilter(); toast('MEMORY UPDATED', 'success'); blip(880,.12);
};
$('btn-reshuffle').onclick = () => {
  memories.forEach((m,i) => { const n = nodes.find(x => x.userData.mid === m.id);
    if (n) n.userData.reshuffleTo = fibPoint(i, memories.length, SPHERE_RADIUS); });
  jlog('RECALIBRATING ORBITAL POSITIONS…'); toast('RESHUFFLING GALAXY', 'info'); blip(440,.2,'triangle');
};
$('btn-wipe').onclick = () => {
  if (!confirm('WIPE ALL MEMORIES? This cannot be undone.')) return;
  memories = []; saveMemories(); rebuildNodes(); deselect();
  toast('ALL MEMORIES WIPED', 'error'); jlog('CORE WIPED — GALAXY EMPTY', 'err');
};
$('btn-shot').onclick = () => {
  composer.render();
  const a = document.createElement('a');
  a.download = 'jarvis-galaxy.png'; a.href = renderer.domElement.toDataURL('image/png'); a.click();
  toast('SNAPSHOT SAVED', 'success'); blip(990,.12);
};
function stopTour() { if (touring) { touring = false; clearInterval(tourTimer); $('btn-tour').classList.remove('on'); jlog('TOUR PAUSED — MANUAL CONTROL'); } }
$('btn-tour').onclick = () => {
  if (touring) return stopTour();
  if (!memories.length) return toast('NO MEMORIES TO TOUR', 'error');
  touring = true; $('btn-tour').classList.add('on'); jlog('AUTO TOUR ENGAGED');
  let i = 0;
  const step = () => { const vis = nodes.filter(n => n.visible); if (!vis.length) return;
    selectMemory(vis[i % vis.length].userData.mid, true); i++; };
  step(); tourTimer = setInterval(step, 3000);
};
$('btn-cine').onclick = () => { document.body.classList.toggle('cine'); blip(600,.12); };
$('btn-sound').onclick = e => { soundOn = !soundOn; e.currentTarget.classList.toggle('on', soundOn); if (soundOn) blip(660,.1); };

addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') { if (e.key !== 'Escape') return; }
  const tabs = ['control','sectors','add','search','system'];
  if (e.key >= '1' && e.key <= '5') document.querySelector(`.dock-btn[data-tab="${tabs[+e.key-1]}"]`).click();
  else if (e.key === 'Escape') { deselect(); cmdInput.blur(); }
  else if (e.key.toLowerCase() === 'h') $('btn-cine').click();
  else if (e.key.toLowerCase() === 'm') $('btn-sound').click();
  else if (e.key.toLowerCase() === 'r') $('btn-rotate-auto').click();
});

/* command bar */
cmdInput.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const raw = cmdInput.value.trim(); if (!raw) return;
  cmdInput.value = ''; handleCommand(raw);
});
function handleCommand(raw) {
  const [cmd, ...rest] = raw.split(' '), arg = rest.join(' ').trim();
  jlog(`CMD: ${raw}`);
  switch (cmd.toLowerCase()) {
    case 'help': jlog('COMMANDS: ask <anything> · find <q> · filter <sector|all> · add <sector>: <text> · tour · spin on/off · bloom on/off · clear'); break;
    case 'ask': {
      if (!arg) return jlog('USAGE: ask <anything> — e.g. ask what is critical?', 'warn');
      if (window.askJarvis) window.askJarvis(arg); else jlog('CHAT LINK OFFLINE — reload the interface', 'err');
      break; }
    case 'find': case 'search': case 'goto': {
      if (!arg) return jlog('SPECIFY QUERY — e.g. find reactor', 'warn');
      const h = memories.find(m => m.title.toLowerCase().includes(arg.toLowerCase()) || m.content.toLowerCase().includes(arg.toLowerCase()));
      if (h) { selectMemory(h.id, true); sndSelect(); } else { jlog(`NO MATCH FOR "${arg}"`, 'warn'); toast('NO MATCH FOUND', 'error'); }
      break; }
    case 'filter': {
      const s = (arg || 'all').toLowerCase();
      const map = { all:'all', personal:'personal', per:'personal', work:'work', wrk:'work', idea:'ideas', ideas:'ideas', crit:'important', critical:'important', important:'important' };
      if (!(s in map)) return jlog('SECTORS: all · personal · work · ideas · critical', 'warn');
      document.querySelector(`.sector-card[data-sector="${map[s]}"]`).click(); break; }
    case 'add': {
      let sector = 'personal', text = arg;
      const m = arg.match(/^(personal|work|ideas?|crit(?:ical)?)\s*:\s*(.+)/i);
      if (m) { const k = m[1].toLowerCase();
        sector = k.startsWith('work') ? 'work' : k.startsWith('ide') ? 'ideas' : k.startsWith('crit') ? 'important' : 'personal';
        text = m[2]; }
      if (!text) return jlog('USAGE: add idea: my brilliant thought', 'warn');
      const mem = { id: 'mem-'+Date.now(), title: text.slice(0,60), content: text, sector, date: new Date().toISOString() };
      memories.push(mem); saveMemories();
      const n = makeNode(mem, randSphere(SPHERE_RADIUS), true);
      const wp = new THREE.Vector3(); n.getWorldPosition(wp); burst(wp, SECTOR[sector].color);
      applyFilter(); selectMemory(mem.id, true); sndStore();
      jlog(`STORED → ${sector.toUpperCase()}`); break; }
    case 'spin': autoRotate = arg !== 'off'; $('btn-rotate-auto').classList.toggle('on', autoRotate); jlog(`AUTO-SPIN ${autoRotate?'ENGAGED':'OFFLINE'}`); break;
    case 'bloom': bloomOn = arg !== 'off'; bloomPass.enabled = bloomOn; $('btn-bloom').classList.toggle('on', bloomOn); break;
    case 'tour': $('btn-tour').click(); break;
    case 'clear': deselect(); searchInput.value=''; cmdInput.value=''; applyFilter(); jlog('FILTERS CLEARED'); break;
    case 'reshuffle': $('btn-reshuffle').click(); break;
    default: {
      // No local match — hand it to the agent instead of guessing.
      if (window.askJarvis) window.askJarvis(raw);
      else jlog('AGENT OFFLINE — reload the interface', 'err');
    }
  }
  blip(700,.08);
}

/* ================= LABELS ================= */
const labelPool = new Map();
function updateLabels() {
  const show = showLabels || hoveredId || selectedId;
  if (!show) { labelLayer.innerHTML = ''; labelPool.clear(); return; }
  const list = showLabels ? nodes.filter(n => n.visible).slice(0, 45)
    : nodes.filter(n => n.visible && (n.userData.mid === hoveredId || n.userData.mid === selectedId));
  const seen = new Set();
  const v = new THREE.Vector3();
  for (const n of list) {
    const m = memories.find(x => x.id === n.userData.mid); if (!m) continue;
    n.getWorldPosition(v); v.project(camera);
    if (v.z > 1 || v.x < -1.1 || v.x > 1.1 || v.y < -1.1 || v.y > 1.1) continue;
    seen.add(n.userData.mid);
    let el = labelPool.get(n.userData.mid);
    if (!el) { el = document.createElement('div'); el.className = 'node-label'; labelLayer.appendChild(el); labelPool.set(n.userData.mid, el); }
    el.textContent = m.title.length > 26 ? m.title.slice(0,26)+'…' : m.title;
    el.style.left = ((v.x*0.5+0.5)*innerWidth)+'px';
    el.style.top = ((-v.y*0.5+0.5)*innerHeight)+'px';
    el.style.borderColor = (SECTOR[m.sector]?.css || '#00e5ff') + '88';
  }
  for (const [id, el] of labelPool) if (!seen.has(id)) { el.remove(); labelPool.delete(id); }
}

/* ================= LOOP ================= */
const clock = new THREE.Clock();
let frames = 0, fpsT = performance.now();
const camFrom = camera.position.clone(), camTo = new THREE.Vector3(0, 3.2, 13.6);
let introT = 0;

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;

  if (!introDone) {
    introT += dt / 3.2;
    const e = 1 - Math.pow(1 - Math.min(1,introT), 3);
    camera.position.lerpVectors(camFrom, camTo, e);
    if (introT >= 1) introDone = true;
  }
  if (autoRotate && introDone) galaxy.rotation.y += spinSpeed * dt * 0.9;
  galaxy.position.y = Math.sin(t*0.55)*0.16;

  coreUniforms.uTime.value = t; shellUniforms.uTime.value = t;
  coreMesh.scale.setScalar(1 + Math.sin(t*2.3)*0.045);
  coreCage.rotation.y += dt*0.5; coreCage.rotation.x += dt*0.22;
  coreCage2.rotation.y -= dt*0.3;
  coreGlow.material.opacity = 0.45 + Math.sin(t*2)*0.12;
  coreLight.intensity = 110 + Math.sin(t*2.3)*25;

  const sp = sparks.geometry.attributes.position;
  for (let i=0;i<sp.count;i++){
    let x = sp.array[i*3], z = sp.array[i*3+2];
    const a = sparkVel[i]*dt*1.4, c = Math.cos(a), s2 = Math.sin(a);
    sp.array[i*3] = x*c - z*s2; sp.array[i*3+2] = x*s2 + z*c;
  }
  sp.needsUpdate = true;

  for (const h of hudRings) {
    if (h.tex) h.tex.offset.x -= dt*h.speed;
    else h.mesh.rotation.z += dt*h.speed*2;
  }
  lat1.rotation.y -= dt*0.02;

  if (dust?.visible) {
    const p = dust.geometry.attributes.position;
    for (let i=0;i<p.count;i++){
      p.array[i*3]   = dustBase[i*3]   + Math.sin(t*0.4+i)*0.18;
      p.array[i*3+1] = dustBase[i*3+1] + Math.cos(t*0.3+i*0.7)*0.18;
      p.array[i*3+2] = dustBase[i*3+2] + Math.sin(t*0.33+i*1.3)*0.18;
    }
    p.needsUpdate = true;
  }

  for (const n of nodes) {
    if (!n.visible) continue;
    if (n.userData.reshuffleTo) {
      n.position.lerp(n.userData.reshuffleTo, 1-Math.pow(0.001, dt));
      if (n.position.distanceTo(n.userData.reshuffleTo) < 0.03) delete n.userData.reshuffleTo;
    }
    const isH = n.userData.mid === hoveredId, isS = n.userData.mid === selectedId;
    n.userData.target = isS ? n.userData.big*1.85 : isH ? n.userData.big*1.55 : n.userData.big*(1+Math.sin(t*3+n.userData.phase)*0.1);
    if (n.scale.x < n.userData.target) n.scale.setScalar(Math.min(n.userData.target, n.scale.x + dt*6));
    else n.scale.setScalar(n.scale.x + (n.userData.target-n.scale.x)*Math.min(1,dt*10));
    n.rotation.y += dt*1.2; n.rotation.x += dt*0.6;
    n.userData.spr.material.opacity = 0.65 + Math.sin(t*2.5+n.userData.phase)*0.2 + (isS?0.25:0);
  }

  if (selectedId) {
    const n = nodes.find(x => x.userData.mid === selectedId);
    if (n?.visible) {
      const wp = new THREE.Vector3(); n.getWorldPosition(wp);
      for (const [h, r] of [[halo,0.42],[halo2,0.58]]) {
        h.position.copy(wp); h.lookAt(camera.position);
        h.scale.setScalar((r/0.42) * (1+Math.sin(t*4)*0.08));
      }
      halo.material.opacity = 0.75+Math.sin(t*4)*0.2;
      const bp = beam.geometry.attributes.position;
      const local = galaxy.worldToLocal(wp.clone());
      bp.setXYZ(0, 0,0,0); bp.setXYZ(1, local.x, local.y, local.z); bp.needsUpdate = true;
      beam.material.opacity = 0.35+Math.sin(t*5)*0.12;
    }
  }
  for (let i = shocks.length-1; i >= 0; i--) {
    const s = shocks[i]; s.t += dt*1.6;
    const k = s.t; s.m.scale.setScalar(1+k*s.max*2); s.m.material.opacity = Math.max(0, 0.9*(1-k));
    s.m.lookAt(camera.position);
    if (k >= 1) { scene.remove(s.m); s.m.geometry.dispose(); s.m.material.dispose(); shocks.splice(i,1); }
  }
  for (let i = bursts.length-1; i >= 0; i--) {
    const b = bursts[i]; b.life -= dt*1.1;
    const p = b.p.geometry.attributes.position;
    for (let j=0;j<p.count;j++){ p.array[j*3]+=b.vel[j*3]*dt; p.array[j*3+1]+=b.vel[j*3+1]*dt; p.array[j*3+2]+=b.vel[j*3+2]*dt;
      b.vel[j*3]*=0.985; b.vel[j*3+1]*=0.985; b.vel[j*3+2]*=0.985; }
    p.needsUpdate = true; b.p.material.opacity = Math.max(0,b.life);
    if (b.life <= 0) { scene.remove(b.p); b.p.geometry.dispose(); b.p.material.dispose(); bursts.splice(i,1); }
  }
  if (linkLines) linkLines.material.opacity = 0.16+Math.sin(t*1.6)*0.05;

  updateLabels();
  controls.update();
  bloomOn ? composer.render() : renderer.render(scene, camera);

  frames++;
  if (performance.now()-fpsT > 1000) {
    fpsEl.textContent = frames; frames = 0; fpsT = performance.now();
    const p = 94 + Math.random()*5; pwrFill.style.width = p+'%'; pwrVal.textContent = p.toFixed(0)+'%';
    $('sys-temp').textContent = (36+Math.random()*2.4).toFixed(1)+'°C';
  }
}

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight);
});

/* clocks */
setInterval(() => {
  timeEl.textContent = new Date().toLocaleTimeString('en-GB');
  const s = ((Date.now()-bootTime)/1000)|0, h = String((s/3600)|0).padStart(2,'0'),
        m = String(((s/60)|0)%60).padStart(2,'0'), ss = String(s%60).padStart(2,'0');
  const u = $('sys-uptime'); if (u) u.textContent = `${h}:${m}:${ss}`;
}, 1000);

/* ================= BOOT ================= */
loadMemories(); rebuildNodes();
const lines = ['MOUNTING NEURAL LATTICE…',`LOADING ${memories.length} MEMORY STARS…`,'CALIBRATING ARC REACTOR…','LINKING SECTORS…','ENGAGING HOLOGRAM…','SYSTEM ONLINE'];
let bi = 0;
const bTimer = setInterval(() => {
  bi++;
  bootFill.style.width = Math.min(100, bi/lines.length*100)+'%';
  bootLog.textContent = lines[Math.min(bi, lines.length-1)];
  blip(300+bi*120, .08, 'square', .03);
  if (bi >= lines.length) {
    clearInterval(bTimer);
    setTimeout(() => {
      bootEl.classList.add('done');
      jlog('J.A.R.V.I.S. ONLINE — memory galaxy ready, sir.');
      jlog('Click a star, type "help", or "ask" me anything.');
      toast('J.A.R.V.I.S. ONLINE', 'success');
    }, 450);
  }
}, 340);
animate();

/* ================= AUDIT TRAIL (what JARVIS did on the PC) ================= */
async function refreshAudit() {
  const box = $('audit-log');
  if (!box) return;
  try {
    const r = await fetch('/api/log');
    if (!r.ok) throw new Error('log ' + r.status);
    const data = await r.json();
    const log = data.log || [];
    box.innerHTML = '';
    if (!log.length) {
      box.innerHTML = '<div class="audit-empty">No computer actions yet this session.</div>';
    } else {
      [...log].reverse().slice(0, 30).forEach(e => {
        const d = document.createElement('div');
        d.className = 'audit-item' + (e.ok ? '' : ' fail');
        const t = document.createElement('span'); t.className = 'a-t'; t.textContent = e.t || '';
        d.appendChild(t);
        d.appendChild(document.createTextNode((e.ok ? '✓ ' : '✗ ') + (e.action || '')));
        box.appendChild(d);
      });
    }
    let mode = box.parentElement?.querySelector('.audit-mode');
    if (!mode) { mode = document.createElement('div'); mode.className = 'audit-mode'; box.after(mode); }
    const req = (data.require_approval || []).join(', ');
    mode.textContent = data.autopilot
      ? (req ? `AUTOPILOT · APPROVAL: ${req}` : 'AUTOPILOT · ALL AUTO')
      : 'APPROVE-EVERY-ACTION';
  } catch (e) {
    if (!box.children.length) box.innerHTML = '<div class="audit-empty">LOG OFFLINE — START BACKEND</div>';
  }
}
$('btn-audit')?.addEventListener('click', () => { refreshAudit(); blip(600, .1); });
$('btn-oskill2')?.addEventListener('click', async () => {
  try { await fetch('/api/os_kill', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }); } catch (e) {}
  toast('STOP SIGNAL SENT', 'info'); refreshAudit();
});
setInterval(refreshAudit, 10000);
setTimeout(refreshAudit, 2500);

/* ================= CHATBOT BRIDGE (Ollama comm-link) ================= */
window.jarvis = {
  getMemories: () => memories.map(m => ({ ...m })),
  storeMemory: (mem) => commitMemoryObject({ ...mem }),
  editMemory: (id, patch) => editMemoryObject(id, patch),
  deleteMemory: (id) => removeMemoryObject(id),
  selectMemory: (id, focus = false) => { revealMemory(id); selectMemory(id, focus); },
  hasMemory: (id) => memories.some(m => m.id === id),
  setFilter: (sector) => setSectorFilter(sector),
  command: (cmd) => galaxyCommand(cmd),
  getState: () => ({
    activeSector, autoRotate, bloomOn, showLabels, touring,
    spin: +spinSpeed.toFixed(2), count: memories.length,
    visible: nodes.filter(n => n.visible).length,
    selected: selectedId,
  }),
};
