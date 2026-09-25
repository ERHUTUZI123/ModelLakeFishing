"""page.py -- turn the built 3D layout into one self-contained WebGL viewer.

Standalone: imports nothing from ModelLakeFishing.  Reads what `build.py`
wrote and emits a single HTML file with the point cloud inlined as base64,
its own orbit camera, and no runtime dependency beyond three.js.

Every drawn point carries two positions, where the trained encoder puts the
model and where the same encoder put it before training, and the page morphs
between them on the GPU.  `#before` in the address opens the untrained view.

    python page.py            ->  lake3d.html, lake3d_offline.html
"""
import base64
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.path.join(REPO, os.environ.get("LAKE3D_WORK",
                                       os.path.join(HERE, "payload")))
OUT_HTML = os.path.join(HERE, "lake3d.html")
OUT_OFFLINE = os.path.join(HERE, "lake3d_offline.html")
THREE_LOCAL = os.path.join(HERE, "three.r128.min.js")

# The plate's palette, so the viewer and the printed figures are one system.
PAPER, INK, SUBINK, FAINT, HAIR = "#ffffff", "#12172a", "#5d6579", "#8b93a6", "#ccd1dc"
AMBER, CRIMSON, GOLD, GOLD_LT = "#d9541b", "#a8264a", "#b0730f", "#edb75a"

# 26 hues that stay apart on white and stay inside the plate's temperature.
# None of them may land on a mark's colour: slot 3 used to be exactly CRIMSON
# and slot 4 exactly GOLD, so bert was painted the query's red and qwen2 the
# gold model's ochre.  Every hue here is at least dE 30 from all five reserved
# tones (crimson, amber, gold, gold light, evidence gold) and dE 19 from its
# nearest neighbour in this list.
FAMILY_COLORS = [
    "#2f3470", "#166f7c", "#3f9e35", "#7c6399", "#4a5bb5", "#2f7d4f",
    "#8a3fa0", "#b46ad0", "#1f6ea8", "#7d6a1f", "#7c9963", "#3f8f86",
    "#5b3fa8", "#c23fb0", "#2a5f9e", "#7a1f4f", "#3d7a2a", "#a0442f",
    "#4f4fa8", "#1f7a63", "#8f2f7a", "#6b7a1f", "#2f6fa8", "#6b4f3a",
    "#5f2f8f", "#2f8f6f",
]
OTHER_COLOR = "#aab1c4"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def build_payload():
    p = np.load(os.path.join(WORK, "packed.npz"))
    meta = json.load(open(os.path.join(WORK, "packed_meta.json"),
                          encoding="utf-8"))
    # typed arrays read the platform's byte order, which is little-endian on
    # everything that runs WebGL, so the payload is written little-endian
    data = {
        "xyz": b64(p["xyz"].astype("<i2")),
        "xyz0": b64(p["xyz0"].astype("<i2")),
        "slot": b64(p["slot"].astype(np.uint8)),
        "ev": b64(p["ev"].astype(np.uint8)),
        "qxyz": b64(p["qxyz"].astype("<i2")),
        "qxyz0": b64(p["qxyz0"].astype("<i2")),
        "lineage": b64(p["lineage"].astype("<u4")),
        "n": int(p["xyz"].shape[0]),
        "nq": int(p["qxyz"].shape[0]),
        "nl": int(p["lineage"].shape[0] // 2),
        "meta": meta,
        "palette": FAMILY_COLORS,
        "other": OTHER_COLOR,
    }
    return data


HTML = r"""<title>Model Lake Galaxy</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root{
    --paper:#ffffff; --ink:#12172a; --sub:#5d6579; --faint:#8b93a6;
    --hair:#ccd1dc; --ghost:#eef1f7; --amber:#d9541b; --crimson:#a8264a;
    --gold:#b0730f; --goldlt:#edb75a; --deep:#2f3470;
    --panel:rgba(255,255,255,.93);
    --rail:302px;                 /* the controls' own column, not an overlay */
    --sans:"IBM Plex Sans","Segoe UI",Calibri,system-ui,-apple-system,Arial,sans-serif;
    --mono:"IBM Plex Mono",Consolas,"SF Mono",Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);
       font:13px/1.5 var(--sans);
       overflow:hidden;-webkit-font-smoothing:antialiased}
  #stage{position:fixed;top:0;right:0;bottom:0;left:var(--rail)}
  canvas{display:block;touch-action:none;width:100%;height:100%}

  .mono{font-family:var(--mono)}

  #rail{position:fixed;left:0;top:0;bottom:0;width:var(--rail);
        border-right:1px solid var(--hair);background:var(--paper);z-index:4}
  #masthead{position:fixed;left:22px;top:18px;pointer-events:none;z-index:5;
            width:calc(var(--rail) - 34px)}
  @media (max-width:760px){ #rail{display:none} }
  #masthead h1{margin:0;font-size:21px;font-weight:600;letter-spacing:-.01em}
  #masthead p{margin:3px 0 0;font-size:11px;color:var(--sub)}

  /* the bottom allowance is the foot's six lines plus a gap, so the panel
     scrolls before it reaches them */
  #panel{position:fixed;left:18px;top:96px;width:266px;max-height:calc(100vh - 208px);
         overflow-y:auto;overflow-x:hidden;background:var(--panel);
         border:1px solid var(--hair);border-radius:9px;padding:12px 13px 14px;
         backdrop-filter:blur(7px);z-index:6}
  #panel::-webkit-scrollbar{width:7px}
  #panel::-webkit-scrollbar-thumb{background:var(--hair);border-radius:4px}
  .sec{margin:0 0 13px}
  .sec:last-child{margin-bottom:0}
  .lab{font-size:9.5px;letter-spacing:.10em;text-transform:uppercase;
       color:var(--faint);margin:0 0 6px;font-weight:600}
  .chips{display:flex;flex-wrap:wrap;gap:4px}
  .chip{border:1px solid var(--hair);background:#fff;color:var(--sub);
        border-radius:5px;padding:3px 8px;font-size:11px;cursor:pointer;
        transition:.13s;font-family:inherit}
  .chip:hover{border-color:var(--faint);color:var(--ink)}
  .chip.on{background:var(--ink);border-color:var(--ink);color:#fff}
  .row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:11.5px;
       color:var(--sub);cursor:pointer;user-select:none}
  .row input{accent-color:var(--deep);margin:0;cursor:pointer}
  .row .sw{width:9px;height:9px;border-radius:50%;flex:none}
  .sl{display:flex;align-items:center;gap:8px;margin:6px 0}
  .sl label{font-size:11px;color:var(--sub);width:52px;flex:none}
  .sl input{flex:1;accent-color:var(--deep);height:2px}
  .sl b{font-size:10px;color:var(--faint);width:30px;text-align:right;
        font-weight:500;font-family:var(--mono)}

  /* before / after is one control with two states, drawn joined so it reads
     as a switch and not as two more chips */
  .seg{display:flex;border:1px solid var(--hair);border-radius:6px;overflow:hidden}
  .seg .chip{flex:1;border:0;border-radius:0;padding:5px 6px;font-size:11.5px}
  .seg .chip + .chip{border-left:1px solid var(--hair)}
  .seg .chip.on{background:var(--ink);color:#fff}
  #spacenote{margin-top:8px;font-size:11px;color:var(--sub);line-height:1.45}
  #spacenote table{width:100%;border-collapse:collapse;margin:8px 0 6px}
  #spacenote th{font-weight:600;font-size:9px;letter-spacing:.08em;
                text-transform:uppercase;color:var(--faint);text-align:right;
                padding:0 0 3px}
  #spacenote td{text-align:right;padding:1.5px 0;font-family:var(--mono);
                font-size:10.5px;color:var(--faint);white-space:nowrap}
  #spacenote th:first-child, #spacenote td:first-child{text-align:left}
  #spacenote td:first-child{font-family:var(--sans);font-size:11px;
                            color:var(--sub);white-space:normal;padding-right:6px}
  #spacenote th.now{color:var(--ink)}
  #spacenote td.now{color:var(--ink);font-weight:500}
  /* how the numbers were measured matters, but not every time the panel is
     opened, so it folds away under the table */
  #spacenote details{font-size:9.5px;color:var(--faint);line-height:1.5}
  #spacenote summary{cursor:pointer;color:var(--sub);font-size:10px;
                     list-style-position:inside}
  #spacenote summary:hover{color:var(--ink)}
  #spacenote details p{margin:4px 0 0}

  #fams{max-height:190px;overflow-y:auto;margin:0 -3px;padding:0 3px}
  #fams::-webkit-scrollbar{width:7px}
  #fams::-webkit-scrollbar-thumb{background:var(--hair);border-radius:4px}
  .fam{display:flex;align-items:center;gap:7px;padding:2.5px 4px;border-radius:4px;
       cursor:pointer;font-size:11.5px;color:var(--sub)}
  .fam:hover{background:var(--ghost);color:var(--ink)}
  .fam.on{background:var(--ink);color:#fff}
  .fam .dot{width:8px;height:8px;border-radius:50%;flex:none}
  .fam .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .fam .ct{font-size:9.5px;color:var(--faint);font-family:var(--mono)}
  .fam.on .ct{color:#b9c0d0}

  #cast{border-top:1px solid var(--hair);padding-top:11px}
  #castinfo{font-size:11px;color:var(--sub);line-height:1.45;margin-top:7px;
            display:none}
  #castinfo b{color:var(--crimson)}
  #castinfo .g{color:var(--gold);font-family:var(--mono);font-size:10.5px;
               word-break:break-all}
  #castinfo .n{display:block;margin-top:5px;color:var(--ink);font-size:10.5px}
  /* ten monospace lines would push the panel past its own height, so the
     list scrolls in place the way the family list does */
  #castinfo .ten{margin-top:6px;border-top:1px solid var(--hair);
                 padding-top:6px;max-height:132px;overflow-y:auto;
                 margin-right:-3px;padding-right:3px}
  #castinfo .ten::-webkit-scrollbar{width:7px}
  #castinfo .ten::-webkit-scrollbar-thumb{background:var(--hair);
                                          border-radius:4px}
  #castinfo .t{display:flex;gap:6px;font-family:var(--mono);font-size:10px;
               color:var(--sub);line-height:1.5;word-break:break-all}
  #castinfo .t i{font-style:normal;color:var(--faint);width:13px;flex:none;
                 text-align:right}
  #castinfo .t.gold{color:var(--gold)}
  #castinfo .t.gold i{color:var(--gold)}

  #hud{position:fixed;right:18px;bottom:16px;text-align:right;z-index:5;
       pointer-events:none}
  #read{display:inline-block;background:var(--panel);border:1px solid var(--hair);
        border-radius:7px;padding:6px 10px;font-size:11.5px;color:var(--ink);
        min-height:29px;min-width:150px;backdrop-filter:blur(7px)}
  #read .k{color:var(--faint);font-size:10px}
  #stat{margin-top:6px;font-size:10px;color:var(--faint);
        font-family:var(--mono)}

  #foot{position:fixed;left:18px;bottom:14px;font-size:9.5px;color:var(--faint);
        font-family:var(--mono);z-index:5;pointer-events:none;
        max-width:300px;line-height:1.55}

  #hint{position:fixed;left:calc(var(--rail) + (100vw - var(--rail))/2);
        bottom:18px;transform:translateX(-50%);
        font-size:11px;color:var(--faint);z-index:5;pointer-events:none;
        transition:opacity .6s;background:var(--panel);padding:5px 12px;
        border-radius:20px;border:1px solid var(--hair)}
  #load{position:fixed;inset:0;display:flex;align-items:center;
        justify-content:center;background:var(--paper);z-index:20;
        flex-direction:column;gap:10px;transition:opacity .5s}
  #load .t{font-size:14px;color:var(--sub)}
  #load .b{width:180px;height:2px;background:var(--ghost);border-radius:2px;
           overflow:hidden}
  #load .b i{display:block;height:100%;width:0;background:var(--deep);
             transition:width .25s}
  .chip:focus-visible, .fam:focus-visible, input:focus-visible{
    outline:2px solid var(--deep); outline-offset:2px; border-radius:5px}
  @media (prefers-reduced-motion: reduce){
    #load, #hint{transition:none}
  }
  @media (max-width:760px){
    :root{--rail:0px}
    #panel{width:210px;top:88px;left:10px;background:var(--panel)}
    /* the rail is gone here, so its width cannot size the masthead: left to
       calc(0px - 34px) it clamped to nothing and stood one word per line */
    #masthead{left:12px;top:12px;right:12px;width:auto}
    #masthead h1{font-size:19px}
    #masthead p{font-size:10.5px;max-width:60vw}
    #foot{display:none}
  }
</style>

<div id="rail"></div>
<div id="stage"></div>

<div id="masthead">
  <h1>The Model Lake</h1>
  <p id="sub"></p>
</div>

<div id="panel">
  <div class="sec">
    <p class="lab">the space</p>
    <div class="seg" id="spaces" role="group" aria-label="the space to show"></div>
    <div id="spacenote"></div>
  </div>
  <div class="sec">
    <p class="lab">colour by</p>
    <div class="chips" id="modes"></div>
  </div>
  <div class="sec">
    <p class="lab">layers</p>
    <label class="row"><input type="checkbox" id="L_models" checked>
      <span class="sw" style="background:#2f3470"></span> the lake</label>
    <label class="row"><input type="checkbox" id="L_queries">
      <span class="sw" style="background:#a8264a"></span>
      <span id="qlab">dataset-task queries</span></label>
    <label class="row"><input type="checkbox" id="L_lineage">
      <span class="sw" style="background:#7f8bca"></span>
      <span id="llab">lineage</span></label>
  </div>
  <div class="sec">
    <p class="lab">view</p>
    <div class="sl"><label>size</label>
      <input type="range" id="S_size" min="0.4" max="6" step="0.1" value="1.9">
      <b id="V_size">1.9</b></div>
    <div class="sl"><label>ink</label>
      <input type="range" id="S_op" min="2" max="60" step="1" value="8">
      <b id="V_op">.08</b></div>
    <div class="sl"><label>haze</label>
      <input type="range" id="S_fog" min="0" max="100" step="1" value="70">
      <b id="V_fog">.70</b></div>
    <div class="sl"><label>spin</label>
      <input type="range" id="S_spin" min="0" max="100" step="1" value="18">
      <b id="V_spin">.18</b></div>
    <div class="chips" style="margin-top:8px">
      <button class="chip" id="B_reset">reset view</button>
      <button class="chip" id="B_top">top down</button>
    </div>
  </div>
  <div class="sec">
    <p class="lab">families <span id="famhint" style="text-transform:none;
      letter-spacing:0;font-weight:400">- click to isolate</span></p>
    <div id="fams"></div>
  </div>
  <div class="sec" id="cast">
    <p class="lab">one cast</p>
    <div class="chips">
      <button class="chip" id="B_cast">show the cast</button>
      <button class="chip" id="B_fly">fly to it</button>
    </div>
    <div id="castinfo"></div>
  </div>
</div>

<div id="hud">
  <div id="read"><span class="k">hover the cloud</span></div>
  <div id="stat"></div>
</div>
<div id="foot"></div>
<div id="hint">drag to orbit &nbsp;·&nbsp; wheel to zoom &nbsp;·&nbsp; right-drag to pan &nbsp;·&nbsp; zoom goes to the cursor</div>
<div id="load"><div class="t">unpacking the lake</div><div class="b"><i id="lb"></i></div></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const DATA = __DATA__;

/* ---------------------------------------------------------------- decode -- */
function bin(s){
  const raw = atob(s), n = raw.length, u = new Uint8Array(n);
  for (let i = 0; i < n; i++) u[i] = raw.charCodeAt(i);
  return u;
}
const lb = document.getElementById('lb');
function step(p){ lb.style.width = (p*100)+'%'; }

const M = DATA.meta;
const N = DATA.n, NQ = DATA.nq, NL = DATA.nl;
const SCALE = 1/32000;
/* the untrained layout comes quantised on its own scale; this puts it in the
   trained layout's units, so the two share one camera */
const BF = M.before;
const SCALE0 = SCALE * BF.scale;

step(.1);
const qi   = new Int16Array(bin(DATA.xyz).buffer);
const qi0  = new Int16Array(bin(DATA.xyz0).buffer);
step(.35);
const slot = bin(DATA.slot);
const evB  = bin(DATA.ev);
const qqi  = new Int16Array(bin(DATA.qxyz).buffer);
const qqi0 = new Int16Array(bin(DATA.qxyz0).buffer);
const lni  = new Uint32Array(bin(DATA.lineage).buffer);
step(.55);

/* every drawn point twice: where the trained encoder puts the model, and
   where the same encoder put it before its first training step */
const posA = new Float32Array(N*3), posB = new Float32Array(N*3);
for (let i = 0; i < N*3; i++){ posA[i] = qi[i]*SCALE; posB[i] = qi0[i]*SCALE0; }
const qpos = new Float32Array(NQ*3), qpos0 = new Float32Array(NQ*3);
for (let i = 0; i < NQ*3; i++){ qpos[i] = qqi[i]*SCALE; qpos0[i] = qqi0[i]*SCALE0; }
/* a lineage link arrives as its two ends' indices among the drawn points, so
   it can be laid down in either space from the points themselves */
const lpos = new Float32Array(NL*6), lpos0 = new Float32Array(NL*6);
for (let k = 0; k < NL*2; k++){
  const i = lni[k]*3;
  for (let d = 0; d < 3; d++){ lpos[k*3+d] = posA[i+d]; lpos0[k*3+d] = posB[i+d]; }
}
const ev = new Uint8Array(N);
for (let i = 0; i < N; i++) ev[i] = (evB[i>>3] >> (7-(i&7))) & 1;
/* what picking and flying read: the space on show, or the one being moved to */
let pos = posA;
step(.75);

/* ------------------------------------------------------------------ scene -- */
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:false,
                                          preserveDrawingBuffer:false});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0xffffff, 1);
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 200);

/* 0 is before training, 1 after.  Every material holds this one object, so a
   single number moves the lake, the queries, the lineage and the cast */
const MIX = {value: 1.0};

const VS = `
  attribute vec4 aColor;                  // rgb + alpha, one 4-byte lane
  attribute vec3 aPos0;                   // the same point before training
  uniform float uSize, uH, uFog, uNear, uFar, uMix;
  varying vec3 vC; varying float vA;
  void main(){
    vec4 mv = modelViewMatrix * vec4(mix(aPos0, position, uMix), 1.0);
    float d = max(-mv.z, 0.02);
    // white paper has no depth cue of its own, so distance is spent as haze
    float f = clamp((d - uNear) / max(uFar - uNear, 1e-4), 0.0, 1.0);
    vC = aColor.rgb;
    vA = aColor.a * (1.0 - uFog * f);
    gl_PointSize = clamp(uSize * uH / d, 0.75, 26.0);
    gl_Position = projectionMatrix * mv;
  }`;
const FS = `
  precision mediump float;
  varying vec3 vC; varying float vA;
  void main(){
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = dot(d, d);
    if (r > 0.25) discard;
    float e = smoothstep(0.25, 0.06, r);
    gl_FragColor = vec4(vC, vA * e);
  }`;

function cloud(position, count, size, position0){
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(position, 3));
  g.setAttribute('aPos0', new THREE.BufferAttribute(position0, 3));
  g.setAttribute('aColor',
    new THREE.BufferAttribute(new Uint8Array(count*4), 4, true));
  const m = new THREE.ShaderMaterial({
    uniforms:{uSize:{value:size}, uH:{value:1100.0}, uFog:{value:0.70},
              uNear:{value:1.0}, uFar:{value:4.0}, uMix:MIX},
    vertexShader:VS, fragmentShader:FS,
    transparent:true, depthTest:true, depthWrite:false,
    blending:THREE.NormalBlending});
  const p = new THREE.Points(g, m);
  p.frustumCulled = false;
  return p;
}

/* lines morph the same way.  Flat colour and no haze, as LineBasicMaterial
   drew them, which cannot take a second position */
const LVS = `
  attribute vec3 aPos0;
  uniform float uMix;
  void main(){
    gl_Position = projectionMatrix * modelViewMatrix
                * vec4(mix(aPos0, position, uMix), 1.0);
  }`;
const LFS = `
  precision mediump float;
  uniform vec3 uColor; uniform float uOpacity;
  void main(){ gl_FragColor = vec4(uColor, uOpacity); }`;
function lines(Kind, position, position0, hex, opacity, depthTest){
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(position, 3));
  g.setAttribute('aPos0', new THREE.BufferAttribute(position0, 3));
  const [r, gr, b] = hex2rgb(hex).map(v=>v/255);
  const o = new Kind(g, new THREE.ShaderMaterial({
    uniforms:{uMix:MIX, uColor:{value:new THREE.Vector3(r, gr, b)},
              uOpacity:{value:opacity}},
    vertexShader:LVS, fragmentShader:LFS,
    transparent:true, depthTest:depthTest, depthWrite:false}));
  o.frustumCulled = false;
  return o;
}

const WS = 0.0012;            // slider unit -> world size
const lake = cloud(posA, N, 1.9*WS, posB);
scene.add(lake);

const queries = cloud(qpos, NQ, 3.4*WS, qpos0);
queries.visible = false;
scene.add(queries);
{ const c = queries.geometry.attributes.aColor.array;
  const [r,g,b] = hex2rgb('#a8264a');
  for (let i=0;i<NQ;i++){ c[i*4]=r; c[i*4+1]=g; c[i*4+2]=b; c[i*4+3]=195; }
  queries.geometry.attributes.aColor.needsUpdate = true; }

const lineage = lines(THREE.LineSegments, lpos, lpos0, '#7f8bca', 0.085, true);
lineage.visible = false;
scene.add(lineage);

/* the plate's prior ramp, so a candidate is the same colour in both.  It is
   declared here, not down with the sprite makers: a const is not hoisted, and
   the cast below paints itself the moment it is built. */
const PRIOR_STOPS = [[0.0,'#a4abbb'],[0.30,'#95a3c1'],
                     [0.60,'#cd9a4c'],[1.0,'#d9541b']]
  .map(([t,h])=>[t, hex2rgb(h)]);
function priorRamp(t){
  t = Math.max(0, Math.min(1, t));
  for (let i=1;i<PRIOR_STOPS.length;i++){
    const [t1,c1] = PRIOR_STOPS[i];
    if (t <= t1 || i === PRIOR_STOPS.length-1){
      const [t0,c0] = PRIOR_STOPS[i-1];
      const f = t1 > t0 ? (t-t0)/(t1-t0) : 0;
      return [c0[0]+(c1[0]-c0[0])*f, c0[1]+(c1[1]-c0[1])*f,
              c0[2]+(c1[2]-c0[2])*f].map(v=>Math.round(v));
    }
  }
  return PRIOR_STOPS[PRIOR_STOPS.length-1][1];
}

/* the cast, read out of the measured seed-0 evaluation: the 1,000 candidates
   HNSW returned, the task prior over exactly those, the ten the system
   answered with, the held-out gold, and the query they were fished for */
const castGroup = new THREE.Group();
castGroup.visible = false;
scene.add(castGroup);
const CAST = (M.casts && M.casts.length) ? M.casts[0] : null;
/* the sprites are a dozen, so they carry both positions and are moved on the
   CPU while the clouds move themselves on the GPU */
const movers = [];
function vec(a, i, s){ return new THREE.Vector3(a[i*3]*s, a[i*3+1]*s, a[i*3+2]*s); }
function mover(o, after, before){ movers.push({o, after, before}); return o; }
function placeMovers(){
  for (const m of movers) m.o.position.lerpVectors(m.before, m.after, MIX.value);
}
let castGoldA = null, castGoldB = null;
if (CAST){
  const pn = CAST.poolXYZ.length/3;
  const pp = new Float32Array(pn*3), pp0 = new Float32Array(pn*3);
  for (let i=0;i<pn*3;i++){ pp[i] = CAST.poolXYZ[i]*SCALE; pp0[i] = CAST.poolXYZ0[i]*SCALE0; }
  const pool = cloud(pp, pn, 5.2*WS, pp0);
  // the pool is painted by the prior the second stage actually read over it,
  // on the plate's own ramp: grey where the prior says nothing, amber where
  // it says most.  A flat amber pool showed which water was dredged but not
  // which part of it the rerank was working from.
  const c = pool.geometry.attributes.aColor.array;
  for (let i=0;i<pn;i++){
    const t = (CAST.poolPrior ? CAST.poolPrior[i] : 255)/255;
    const [r1,g1,b1] = priorRamp(t);
    c[i*4]=r1; c[i*4+1]=g1; c[i*4+2]=b1;
    c[i*4+3]=Math.round(150 + 105*Math.pow(t, 1.4));
  }
  pool.geometry.attributes.aColor.needsUpdate = true;
  castGroup.add(pool);

  castGoldA = vec(CAST.goldXYZ, 0, SCALE);
  castGoldB = vec(CAST.goldXYZ0, 0, SCALE0);
  const qvA = vec(CAST.queryXYZ, 0, SCALE), qvB = vec(CAST.queryXYZ0, 0, SCALE0);

  // the ten the system returned, ringed.  The one that is the gold is left to
  // the star rather than ringed twice, so each of the ten is drawn once.
  const ring = new THREE.SpriteMaterial({map:ringTex(), transparent:true,
                                         depthTest:false, depthWrite:false});
  const tn = CAST.topXYZ ? CAST.topXYZ.length/3 : 0;
  for (let i=0;i<tn;i++){
    if (i+1 === CAST.goldPosition) continue;     // that one is the star
    const s = new THREE.Sprite(ring);
    s.scale.setScalar(0.020);
    castGroup.add(mover(s, vec(CAST.topXYZ, i, SCALE), vec(CAST.topXYZ0, i, SCALE0)));
  }

  const star = new THREE.Sprite(new THREE.SpriteMaterial({
    map:starTex(), transparent:true, depthTest:false, depthWrite:false}));
  star.scale.setScalar(0.040);
  castGroup.add(mover(star, castGoldA, castGoldB));
  const cross = new THREE.Sprite(new THREE.SpriteMaterial({
    map:crossTex(), transparent:true, depthTest:false, depthWrite:false}));
  cross.scale.setScalar(0.026);
  castGroup.add(mover(cross, qvA, qvB));
  castGroup.add(lines(THREE.Line,
    new Float32Array([qvA.x, qvA.y, qvA.z, castGoldA.x, castGoldA.y, castGoldA.z]),
    new Float32Array([qvB.x, qvB.y, qvB.z, castGoldB.x, castGoldB.y, castGoldB.z]),
    '#b0730f', 0.85, false));
  placeMovers();
}

function sprite(draw){
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const x = c.getContext('2d'); draw(x);
  const t = new THREE.CanvasTexture(c); t.needsUpdate = true; return t;
}
function starTex(){
  return sprite(x=>{
    x.translate(64,64); x.beginPath();
    for (let i=0;i<10;i++){
      const a = -Math.PI/2 + i*Math.PI/5, r = i%2 ? 23 : 55;
      x[i?'lineTo':'moveTo'](Math.cos(a)*r, Math.sin(a)*r);
    }
    x.closePath(); x.fillStyle='#edb75a'; x.fill();
    x.lineWidth=7; x.strokeStyle='#12172a'; x.stroke();
  });
}
function crossTex(){
  return sprite(x=>{
    x.fillStyle='#a8264a'; const w=22, L=56;
    x.fillRect(64-w/2,64-L/2,w,L); x.fillRect(64-L/2,64-w/2,L,w);
    x.strokeStyle='#ffffff'; x.lineWidth=6;
    x.strokeRect(64-w/2,64-L/2,w,L); x.strokeRect(64-L/2,64-w/2,L,w);
  });
}
/* one of the ten returned: the plate draws these as numbered discs, which at
   this scale would be unreadable, so here they keep the disc and drop the
   number -- the panel lists the ten in order instead */
function ringTex(){
  return sprite(x=>{
    x.beginPath(); x.arc(64,64,38,0,Math.PI*2);
    x.fillStyle='#ffffff'; x.fill();
    x.lineWidth=13; x.strokeStyle='#12172a'; x.stroke();
  });
}
function hex2rgb(h){
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16),
          parseInt(h.slice(5,7),16)];
}
step(.9);

/* ------------------------------------------------------------- colouring -- */
const PAL = DATA.palette.map(hex2rgb);
const OTHER = hex2rgb(DATA.other);
const RAMP = ['#d8dff3','#adb9e4','#7f8bca','#535ca9','#343783','#15163c']
             .map(hex2rgb);
const GOLDC = hex2rgb('#c98a1e'), PALE = hex2rgb('#c9cedd');

let mode = 'depth', highlight = -1, ink = 0.08;
/* depth is read in the trained lake and then stays with the model, like every
   other colour, so the untrained view shows where the lake's core came from */
const rad = new Float32Array(N);
for (let i=0;i<N;i++){
  rad[i] = Math.sqrt(posA[i*3]**2 + posA[i*3+1]**2 + posA[i*3+2]**2);
}
let radMax = 0;
for (let i=0;i<N;i++) if (rad[i] > radMax) radMax = rad[i];

function ramp(t){
  // the core is already the densest thing on screen; letting it also take the
  // darkest end of the ramp turns it into a silhouette, so the ramp stops short
  t = Math.pow(Math.max(0, Math.min(1, t)), 1.55) * 0.80;
  t = Math.max(0, Math.min(0.999, t)) * (RAMP.length-1);
  const i = Math.floor(t), f = t-i, a = RAMP[i], b = RAMP[Math.min(i+1, RAMP.length-1)];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}

function paint(){
  const C = lake.geometry.attributes.aColor.array;
  const base = Math.round(ink*255);
  for (let i=0;i<N;i++){
    let c, a = base;
    if (mode === 'family'){
      // fewer than half the points belong to a named family; the rest have to
      // stay legible as context without smothering the ones that are named
      if (slot[i]){ c = PAL[slot[i]-1]; a = Math.min(255, Math.round(base*1.5)); }
      else        { c = OTHER;          a = Math.round(base*0.42); }
    } else if (mode === 'evidence'){
      // evidence is rare -- 1.5% of what is drawn -- so it is shown at full
      // strength over a ghost of the lake rather than at the same weight
      if (ev[i]){ c = GOLDC; a = 255; }
      else      { c = PALE;  a = Math.round(base*0.22); }
    } else if (mode === 'depth'){
      c = ramp(1 - rad[i]/radMax);
    } else {
      c = RAMP[4];
    }
    if (highlight >= 0){
      if (slot[i] !== highlight){ c = PALE; a = Math.round(base*0.16); }
      else a = Math.min(255, Math.round(base*2.1));
    }
    C[i*4] = c[0]|0; C[i*4+1] = c[1]|0; C[i*4+2] = c[2]|0; C[i*4+3] = a;
  }
  lake.geometry.attributes.aColor.needsUpdate = true;
}

/* ------------------------------------------------------------- the camera -- */
const CALM = typeof matchMedia === 'function' &&
  matchMedia('(prefers-reduced-motion: reduce)').matches;
let dist = 1.50, theta = 0.6, phi = 1.16, spin = CALM ? 0 : 0.18;
const target = new THREE.Vector3(0,0,0);
let tw = null;

function place(){
  const s = Math.sin(phi), x = dist*s*Math.sin(theta),
        y = dist*Math.cos(phi), z = dist*s*Math.cos(theta);
  camera.position.set(target.x+x, target.y+y, target.z+z);
  camera.lookAt(target);
}
function flyTo(p, d, ms){
  const t0 = performance.now();
  const s = {t:target.clone(), d:dist};
  tw = (now)=>{
    let u = Math.min(1, (now-t0)/ms);
    u = u<0.5 ? 4*u*u*u : 1-Math.pow(-2*u+2,3)/2;
    target.lerpVectors(s.t, p, u);
    dist = s.d + (d-s.d)*u;
    if (u >= 1) tw = null;
  };
}

const el = renderer.domElement;
let drag = 0, lx = 0, ly = 0, pid = null;
const TARGET_MAX = 2.4;              // the pivot stays inside the lake

// A capture that is never released is how the orbit centre used to wander off:
// one right-drag released outside the window left the drag latched, and every
// later mouse move went on panning the pivot.
function endDrag(){
  drag = 0;
  if (pid !== null){
    try { if (el.hasPointerCapture && el.hasPointerCapture(pid))
            el.releasePointerCapture(pid); } catch (_) {}
  }
  pid = null;
}
el.addEventListener('pointerdown', e=>{
  drag = (e.button === 2 || e.shiftKey) ? 2 : 1;
  lx = e.clientX; ly = e.clientY; pid = e.pointerId;
  try { el.setPointerCapture(pid); } catch (_) {}
  hideHint();
});
el.addEventListener('pointerup', endDrag);
el.addEventListener('pointercancel', endDrag);
el.addEventListener('lostpointercapture', endDrag);
el.addEventListener('pointerleave', e=>{ if (!e.buttons) endDrag(); });
addEventListener('blur', endDrag);
addEventListener('pointerup', endDrag);
el.addEventListener('pointermove', e=>{
  if (drag){
    const dx = e.clientX-lx, dy = e.clientY-ly; lx = e.clientX; ly = e.clientY;
    if (drag === 1){
      theta -= dx*0.0052;
      phi = Math.max(0.10, Math.min(Math.PI-0.10, phi - dy*0.0052));
    } else {
      const f = dist*0.0016;
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix,0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix,1);
      target.addScaledVector(right, -dx*f).addScaledVector(up, dy*f);
      if (target.length() > TARGET_MAX) target.setLength(TARGET_MAX);
    }
    tw = null;
  } else queueHover(e.clientX, e.clientY);
});
/* ---- zoom about whatever is under the cursor ---------------------------- */
const ORIGIN = new THREE.Vector3(0,0,0);
const _v = new THREE.Vector3(), _dir = new THREE.Vector3(), _n = new THREE.Vector3();
let anchor = new THREE.Vector3(), anchorAt = 0, anchorX = 1e9, anchorY = 1e9;

function ray(mx, my){
  const r = el.getBoundingClientRect();
  const nx = ((mx-r.left)/r.width)*2 - 1, ny = -((my-r.top)/r.height)*2 + 1;
  _v.set(nx, ny, 0.5).unproject(camera);
  return _dir.copy(_v).sub(camera.position).normalize();
}

function anchorFor(mx, my){
  // one pass over the cloud is too much for every wheel tick, so the anchor is
  // resolved once per gesture and held until the cursor actually moves
  const now = performance.now();
  const moved = Math.abs(mx-anchorX) > 6 || Math.abs(my-anchorY) > 6;
  if (!moved && now - anchorAt < 220){ anchorAt = now; return anchor; }
  anchorX = mx; anchorY = my; anchorAt = now;

  const d = ray(mx, my);
  const hit = lake.visible ? nearestPoint(mx, my, 0.0016) : -1;
  if (hit >= 0){
    anchor.set(pos[hit*3], pos[hit*3+1], pos[hit*3+2]);
    return anchor;
  }
  // nothing under the cursor: fall back to the plane through the orbit centre
  _n.copy(target).sub(camera.position).normalize();
  const denom = d.dot(_n);
  const t = Math.abs(denom) < 1e-4
    ? camera.position.distanceTo(target)
    : _v.copy(target).sub(camera.position).dot(_n) / denom;
  anchor.copy(camera.position).addScaledVector(d, Math.max(t, 0.02));
  return anchor;
}

function zoomAt(mx, my, k){
  const before = dist;
  dist = Math.max(0.05, Math.min(24, dist * k));
  k = dist / before;                       // the clamped factor, so A holds
  if (k === 1) return;
  const A = anchorFor(mx, my);
  target.set(A.x + (target.x-A.x)*k,
             A.y + (target.y-A.y)*k,
             A.z + (target.z-A.z)*k);
  // pulling all the way out drifts back to the middle rather than stranding
  // the view on whatever was last under the cursor
  if (dist > 2.4) target.lerp(ORIGIN, Math.min(0.12, (dist-2.4)*0.10));
  if (target.length() > TARGET_MAX) target.setLength(TARGET_MAX);
  tw = null;
}

el.addEventListener('wheel', e=>{
  e.preventDefault();
  zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY*0.0011));
  hideHint();
}, {passive:false});
el.addEventListener('contextmenu', e=>e.preventDefault());
el.addEventListener('dblclick', e=>{
  const hit = nearestPoint(e.clientX, e.clientY, 0.004);
  const p = hit >= 0
    ? new THREE.Vector3(pos[hit*3], pos[hit*3+1], pos[hit*3+2])
    : new THREE.Vector3(0, 0, 0);
  flyTo(p, dist, 520);
});

/* pinch on touch */
let pinch = 0;
el.addEventListener('touchstart', e=>{
  if (e.touches.length === 2){
    pinch = Math.hypot(e.touches[0].clientX-e.touches[1].clientX,
                       e.touches[0].clientY-e.touches[1].clientY);
  }
}, {passive:true});
el.addEventListener('touchmove', e=>{
  if (e.touches.length === 2 && pinch){
    const d = Math.hypot(e.touches[0].clientX-e.touches[1].clientX,
                         e.touches[0].clientY-e.touches[1].clientY);
    zoomAt((e.touches[0].clientX + e.touches[1].clientX)/2,
           (e.touches[0].clientY + e.touches[1].clientY)/2, pinch/d);
    pinch = d;
  }
}, {passive:true});

/* --------------------------------------------------------------- picking -- */
const readEl = document.getElementById('read');
let hoverPend = null, hoverAt = 0;
function queueHover(x, y){ hoverPend = [x,y]; }

const vp = new THREE.Matrix4();

/* nearest drawn point to a screen position, in squared NDC distance */
let morphing = false;
function nearestPoint(mx, my, tol){
  if (morphing) return -1;           // the points are between their two places
  const r = el.getBoundingClientRect();
  const nx = ((mx-r.left)/r.width)*2 - 1, ny = -((my-r.top)/r.height)*2 + 1;
  vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = vp.elements;
  let best = -1, bd = tol;
  for (let i=0;i<N;i++){
    if (highlight >= 0 && slot[i] !== highlight) continue;
    const x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    const w = e[3]*x + e[7]*y + e[11]*z + e[15];
    if (w <= 0.02) continue;
    const cx = (e[0]*x + e[4]*y + e[8]*z + e[12])/w;
    if (cx < -1 || cx > 1) continue;
    const cy = (e[1]*x + e[5]*y + e[9]*z + e[13])/w;
    if (cy < -1 || cy > 1) continue;
    const dx = cx-nx, dy = cy-ny, d = dx*dx + dy*dy;
    if (d < bd){ bd = d; best = i; }
  }
  return best;
}

function doHover(){
  if (!hoverPend) return;
  const now = performance.now();
  if (now - hoverAt < 90) return;
  hoverAt = now;
  const [mx, my] = hoverPend; hoverPend = null;
  if (!lake.visible){ readEl.innerHTML = '<span class="k">the lake is hidden</span>'; return; }
  const best = nearestPoint(mx, my, 0.00055);
  if (best < 0){ readEl.innerHTML = '<span class="k">hover the cloud</span>'; return; }
  const fam = slot[best] ? M.families[slot[best]-1].name : 'unnamed family';
  readEl.innerHTML =
    '<b>'+esc(fam)+'</b><br><span class="k">'+
    (ev[best] ? 'carries evaluation evidence' : 'no evidence, found by representation alone')+
    '</span>';
}
function esc(s){ return String(s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

/* ------------------------------------------------------------------- ui -- */
const MODES = [['depth','depth'],['family','families'],
               ['evidence','evidence'],['plain','plain']];
const modesEl = document.getElementById('modes');
MODES.forEach(([k,label])=>{
  const b = document.createElement('button');
  b.className = 'chip' + (k===mode?' on':''); b.textContent = label;
  b.onclick = ()=>{ mode = k; paint();
    [...modesEl.children].forEach(c=>c.classList.toggle('on', c===b)); };
  modesEl.appendChild(b);
});

const famsEl = document.getElementById('fams');
M.families.forEach((f,i)=>{
  const d = document.createElement('div');
  d.className = 'fam'; d.tabIndex = 0;
  d.onkeydown = e=>{ if (e.key === 'Enter' || e.key === ' '){ e.preventDefault(); d.onclick(); } };
  d.innerHTML = '<span class="dot" style="background:'+DATA.palette[i]+'"></span>'+
    '<span class="nm">'+esc(f.name)+'</span><span class="ct">'+
    f.count.toLocaleString()+'</span>';
  d.onclick = ()=>{
    const on = highlight === i+1;
    highlight = on ? -1 : i+1;
    [...famsEl.children].forEach((c,j)=>c.classList.toggle('on', !on && j===i));
    paint();
    if (!on) flyToSlot(i+1);
  };
  famsEl.appendChild(d);
});
function flyToSlot(s){
  let cx=0, cy=0, cz=0, n=0;
  for (let i=0;i<N;i++) if (slot[i]===s){ cx+=pos[i*3]; cy+=pos[i*3+1]; cz+=pos[i*3+2]; n++; }
  if (!n) return;
  let sp = 0;
  const c = new THREE.Vector3(cx/n, cy/n, cz/n);
  for (let i=0;i<N;i++) if (slot[i]===s){
    sp = Math.max(sp, (pos[i*3]-c.x)**2 + (pos[i*3+1]-c.y)**2 + (pos[i*3+2]-c.z)**2);
  }
  flyTo(c, Math.max(0.35, Math.sqrt(sp)*2.3), 900);
}

function bindSlider(id, out, fn, fmt){
  const s = document.getElementById(id), o = document.getElementById(out);
  const run = ()=>{ const v = parseFloat(s.value); o.textContent = fmt(v); fn(v); };
  s.oninput = run; run();
}
bindSlider('S_size','V_size', v=>{
  lake.material.uniforms.uSize.value = v*WS;
  queries.material.uniforms.uSize.value = v*2.0*WS;
}, v=>v.toFixed(1));
bindSlider('S_op','V_op', v=>{ ink = v/100; paint(); },
  v=>(v/100).toFixed(2).slice(1));
bindSlider('S_fog','V_fog', v=>{
  [lake, queries].forEach(o=>o.material.uniforms.uFog.value = v/100);
  castGroup.children.forEach(c=>{
    if (c.material && c.material.uniforms && c.material.uniforms.uFog)
      c.material.uniforms.uFog.value = (v/100)*0.5;
  });
}, v=>(v/100).toFixed(2).slice(1));
if (CALM) document.getElementById('S_spin').value = '0';
bindSlider('S_spin','V_spin', v=>{ spin = v/100; }, v=>(v/100).toFixed(2).slice(1));

document.getElementById('L_models').onchange = e=>{ lake.visible = e.target.checked; };
document.getElementById('L_queries').onchange = e=>{ queries.visible = e.target.checked; };
document.getElementById('L_lineage').onchange = e=>{ lineage.visible = e.target.checked; };
document.getElementById('B_reset').onclick = ()=>{
  theta = 0.6; phi = 1.16; flyTo(new THREE.Vector3(0,0,0), 1.50, 700); };
document.getElementById('B_top').onclick = ()=>{
  phi = 0.13; theta = 0.0; flyTo(new THREE.Vector3(0,0,0), 1.45, 700); };

const castInfo = document.getElementById('castinfo');
document.getElementById('B_cast').onclick = e=>{
  castGroup.visible = !castGroup.visible;
  e.target.classList.toggle('on', castGroup.visible);
  castInfo.style.display = castGroup.visible ? 'block' : 'none';
};
document.getElementById('B_fly').onclick = ()=>{
  // Pull back far enough to hold the ten the system returned, whatever their
  // spread.  The floor is not arbitrary: the star and the rings are sprites
  // of a fixed size in the world, so past a point closing in only makes the
  // marks enormous and shows less -- 0.42 is where the star reads as a mark
  // rather than a wall.  A cast scattered across the lake needs more.  In the
  // untrained space the ten are scattered far wider, and the pull-back shows it.
  if (!castGoldA) return;
  const T = space ? CAST.topXYZ : CAST.topXYZ0, s = space ? SCALE : SCALE0;
  const g = space ? castGoldA : castGoldB;
  let r = 0;
  for (let i = 0; i < (T ? T.length/3 : 0); i++){
    r = Math.max(r, Math.hypot(T[i*3]*s - g.x, T[i*3+1]*s - g.y, T[i*3+2]*s - g.z));
  }
  flyTo(g.clone(), Math.max(0.42, Math.min(2.4, r * 3.0)), 1100);
};
const ord = n => n.toLocaleString() + (n%100>=11 && n%100<=13 ? 'th'
                      : ({1:'st',2:'nd',3:'rd'}[n%10] || 'th'));
const EV = BF.eval;
function castNote(){
  if (!CAST) return;
  const ten = (CAST.topNames||[]).map((nm,i)=>
    '<span class="t'+(nm === CAST.gold ? ' gold' : '')+'">'+
    '<i>'+(i+1)+'</i>'+esc(nm)+'</span>').join('');
  // Before training nothing was retrieved: the untrained encoder never ran as
  // a system.  What that view shows is where the same models sat, and the one
  // measured fact about them there is how far down the gold ranked.
  castInfo.innerHTML =
    '<b>'+esc(CAST.label)+'</b><br>'+
    (space
      ? 'the '+CAST.poolN.toLocaleString()+' candidates HNSW returned, painted by '+
        'the task prior read over them; the ten the system answered with ringed, '+
        'the held-out best model as the gold star, the query as the crimson '+
        'cross.<br>'+
        '<span class="n">gold ' + (CAST.denseRank
          ? ord(CAST.denseRank)+' of '+CAST.poolN.toLocaleString()+' on cosine '+
            'alone, returned '+ord(CAST.goldPosition)+' after the prior'
          : 'never reached the pool') + '</span>'
      : 'the same '+CAST.poolN.toLocaleString()+' candidates, the same ten and '+
        'the same gold, where the untrained encoder put them; the crimson cross '+
        'sits among this query\'s untrained top 64.<br>'+
        '<span class="n">before training the gold was '+
        ord(EV.castDenseRank.before)+' of '+M.nModels.toLocaleString()+
        ' on cosine, nowhere near a first stage of 1,000</span>') +
    '<div class="ten">'+ten+'</div>';
}

/* ---------------------------------------------------------- the two spaces -- */
const subEl = document.getElementById('sub');
const SUB = [
  'the same ' + M.nShown.toLocaleString() + ' models, where the encoder put ' +
  'them before its first training step',
  M.nModels.toLocaleString() + ' models in one 128-dimensional retrieval space, ' +
  M.nShown.toLocaleString() + ' of them drawn here'];
const spacesEl = document.getElementById('spaces');
const spaceNoteEl = document.getElementById('spacenote');
let space = 1, mixTw = null;
const spaceBtns = ['before training', 'after training'].map((label, k)=>{
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = label;
  b.onclick = ()=>setSpace(k);
  b.setAttribute('aria-pressed', 'false');
  spacesEl.appendChild(b);
  return b;
});

function num(v){
  return Number.isInteger(v) ? v.toLocaleString()
       : v.toLocaleString(undefined, {minimumFractionDigits:1, maximumFractionDigits:1});
}
function spaceNote(){
  // both columns come from the same evaluator over both encoders, and the
  // table is only shown if, on the trained one, it reproduced the A0 record
  const c = k => k === space ? ' class="now"' : '';
  const row = (label, b, a) =>
    '<tr><td>'+label+'</td><td'+c(0)+'>'+b+'</td><td'+c(1)+'>'+a+'</td></tr>';
  const open = spaceNoteEl.querySelector && spaceNoteEl.querySelector('details');
  spaceNoteEl.innerHTML =
    (space
      ? 'The encoder after its 25 training epochs. Before training shows it at '
      : 'The encoder A0 trained, at ') +
    'the weights it started from (init seed ' + BF.initSeed + '), on the same ' +
    'graph and the same rows. Colours travel with each model.' +
    (EV.reproducesRecord
      ? '<table><tr><th></th><th'+c(0)+'>before</th><th'+c(1)+'>after</th></tr>' +
        row('gold@10', EV.before.gold10.toFixed(3), EV.after.gold10.toFixed(3)) +
        row('gold in first 1,000', num(EV.before.within1000), num(EV.after.within1000)) +
        row('median gold rank', num(EV.before.denseMedian), num(EV.after.denseMedian)) +
        '</table><details' + (open && open.open ? ' open' : '') +
        '><summary>how these were measured</summary><p>' +
        EV.nQueries.toLocaleString() + ' held-out queries, split seed ' +
        M.source.splitSeed + '. The first stage takes the exact top 1,000 by ' +
        'cosine and the task prior reranks them; the rank is by cosine among all ' +
        M.nModels.toLocaleString() + '. The same A0 evaluator scored both ' +
        'encoders; on the trained one it reproduces the recorded scores ' +
        'exactly.</p><p>Nothing was retrieved in the untrained space: its ' +
        'layout is the same PCA and UMAP recipe, over the same anchors, turned ' +
        'rigidly onto the trained one.</p></details>'
      : '');
}

/* One number moves everything.  The picking positions jump to the target at
   once -- hovering is off until the points arrive -- so a family chosen
   mid-flight is flown to where it is going, not where it was. */
function setSpace(s, instant){
  s = s ? 1 : 0;
  if (s === space && !mixTw) return;
  space = s;
  pos = s ? posA : posB;
  const from = MIX.value, t0 = performance.now();
  const ms = (instant || CALM) ? 0 : 1800 * Math.abs(s - from);
  morphing = ms > 0;
  mixTw = (now)=>{
    let u = ms ? Math.min(1, (now - t0)/ms) : 1;
    u = u<0.5 ? 4*u*u*u : 1-Math.pow(-2*u+2,3)/2;
    MIX.value = from + (s - from)*u;
    placeMovers();
    if (u >= 1){ mixTw = null; morphing = false; }
  };
  if (!ms) mixTw(t0);
  showSpace(!instant);
}
function showSpace(address){
  spaceBtns.forEach((b, k)=>{
    b.classList.toggle('on', k === space);
    b.setAttribute('aria-pressed', k === space ? 'true' : 'false');
  });
  subEl.textContent = SUB[space];
  spaceNote();
  castNote();
  // once someone switches, the address says which space is up, so a copied
  // link opens on it; a page that was only opened is left as it was opened
  if (address && typeof history !== 'undefined' && history.replaceState){
    try { history.replaceState(null, '', space ? '#after' : '#before'); } catch (_) {}
  }
}

document.getElementById('qlab').textContent =
  M.nQueries.toLocaleString() + ' dataset-task queries';
document.getElementById('llab').textContent =
  M.nLineageEdges.toLocaleString() + ' lineage links';
document.getElementById('foot').innerHTML =
  'PCA ' + M.umap.pca_dim + ' of z_m, UMAP over ' + M.anchors.toLocaleString() +
  ' anchors,<br>every other row by ' + M.umap.knn + '-anchor interpolation<br>' +
  M.nEvidence.toLocaleString() + ' models carry evaluation evidence<br>' +
  (M.source
    ? M.source.run + ', split seed ' + M.source.splitSeed + '<br>graph ' +
      M.source.graphDigest.slice(0,14) + '…'
    : '') +
  '<br>before: init seed ' + BF.initSeed + ', PCA ' + BF.pcaDim + ', same anchors';

let hintGone = false;
function hideHint(){
  if (hintGone) return; hintGone = true;
  const h = document.getElementById('hint');
  h.style.opacity = 0; setTimeout(()=>h.remove(), 700);
}
setTimeout(hideHint, 9000);

/* ----------------------------------------------------------------- loop -- */
function resize(){
  // size to the stage, not the window: the panel has its own column now, so
  // the middle of the canvas is the middle of what the eye reads, and the
  // orbit centre sits there without any correction
  const r = stage.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  renderer.setSize(w, h);          // updateStyle stays on: without the CSS size
                                   // the canvas lays out at w * devicePixelRatio
                                   // and the picture drifts to the bottom-right
  camera.aspect = w/h;
  camera.updateProjectionMatrix();
  const px = h / (2*Math.tan(camera.fov*Math.PI/360));
  lake.material.uniforms.uH.value = px;
  queries.material.uniforms.uH.value = px;
  castGroup.children.forEach(c=>{
    if (c.material && c.material.uniforms && c.material.uniforms.uH)
      c.material.uniforms.uH.value = px;
  });
}
addEventListener('resize', resize);
if (typeof ResizeObserver === 'function') new ResizeObserver(resize).observe(stage);

const statEl = document.getElementById('stat');
let frames = 0, last = performance.now();
function tick(now){
  requestAnimationFrame(tick);
  if (tw) tw(now);
  if (mixTw) mixTw(now);
  if (spin > 0 && !drag) theta += spin*0.0016;
  place();
  const near = dist*0.42, far = dist*1.85;
  [lake, queries].forEach(o=>{
    o.material.uniforms.uNear.value = near;
    o.material.uniforms.uFar.value = far;
  });
  castGroup.children.forEach(c=>{
    if (c.material && c.material.uniforms && c.material.uniforms.uNear){
      c.material.uniforms.uNear.value = near;
      c.material.uniforms.uFar.value = far;
    }
  });
  renderer.render(scene, camera);
  doHover();
  frames++;
  if (now - last > 1000){
    statEl.textContent = Math.round(frames*1000/(now-last)) + ' fps  ·  ' +
      N.toLocaleString() + ' points';
    frames = 0; last = now;
  }
}

paint(); resize(); place(); step(1);
if (typeof location !== 'undefined' && /^#before$/i.test(location.hash)) setSpace(0, true);
else showSpace(false);
requestAnimationFrame(tick);
setTimeout(()=>{
  const l = document.getElementById('load');
  l.style.opacity = 0; setTimeout(()=>l.remove(), 600);
}, 260);
</script>
"""


CDN_TAG = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/'
           'r128/three.min.js"></script>')
FONT_TAG = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
            'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:'
            'wght@400;500&display=swap">')


def _comment_spans(src, css=False):
    """(start, end) of every comment in a script or stylesheet body.

    A small scanner rather than a regex, because the script holds strings, a
    regular expression and GLSL in template literals -- whose own `//` notes
    are shader source, not page comments, and stay.
    """
    spans, i, n = [], 0, len(src)
    prev = ""                                   # last non-space character
    while i < n:
        ch, nx = src[i], src[i + 1] if i + 1 < n else ""
        if ch == "/" and nx == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            spans.append((i, j))
            i = j
            continue
        if not css and ch == "/" and nx == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            spans.append((i, j))
            i = j
            continue
        if ch in "'\"" or (not css and ch == "`"):
            j = i + 1
            while j < n and src[j] != ch:
                j += 2 if src[j] == "\\" else 1
            i, prev = j + 1, ch
            continue
        if not css and ch == "/" and (prev == "" or prev in "(,=:[!&|?{};+-*%<>~^"):
            j, klass = i + 1, False              # a regular-expression literal
            while j < n and src[j] != "\n":
                c = src[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "[":
                    klass = True
                elif c == "]":
                    klass = False
                elif c == "/" and not klass:
                    break
                j += 1
            i, prev = j + 1, "/"
            continue
        if not ch.isspace():
            prev = ch
        i += 1
    return spans


def _strip_block(src, css=False):
    """Drop comments line by line, the way the release tree dropped them.

    A comment after code goes with the spaces before it.  A run of lines that
    held nothing but comment becomes one blank line, or nothing when the line
    above is already blank.
    """
    inside = bytearray(len(src))
    for a, b in _comment_spans(src, css):
        inside[a:b] = b"\x01" * (b - a)
    out, at = [], 0
    for line in src.split("\n"):
        mask = inside[at:at + len(line)]
        at += len(line) + 1
        if not any(mask):
            out.append(line)
            continue
        code = "".join(c for c, m in zip(line, mask) if not m)
        if code.strip():
            out.append(code.rstrip())
        elif not out or out[-1].strip():
            out.append("")
    return "\n".join(out)


def strip_page_comments(html):
    """The page with its own stylesheet and script comments removed.

    Only the first <style> and the last bare <script> are touched: that is the
    page's own code.  The offline build inlines three.js in an earlier
    <script>, and its licence header goes wherever the library goes.
    """
    m = re.search(r"<style>(.*?)</style>", html, flags=re.S)
    html = html[:m.start(1)] + _strip_block(m.group(1), css=True) + html[m.end(1):]
    m = list(re.finditer(r"<script>(.*?)</script>", html, flags=re.S))[-1]
    return html[:m.start(1)] + _strip_block(m.group(1)) + html[m.end(1):]


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return os.path.getsize(path) / 2 ** 20


def _offline(html):
    """Three.js inlined and the web fonts dropped back to what the machine has."""
    lib = open(THREE_LOCAL, encoding="utf-8").read()
    off = (html.replace(CDN_TAG, "<script>" + lib + "</script>")
               .replace(FONT_TAG, ""))
    assert CDN_TAG not in off and "fonts.googleapis" not in off
    return off


def main():
    data = build_payload()
    blob = json.dumps(data, separators=(",", ":"))
    # Match the shipped pages; leave the payload and third-party library intact.
    html = strip_page_comments(HTML).replace("__DATA__", blob)
    mb = _write(OUT_HTML, html)
    print("wrote %s  (%.2f MB)" % (OUT_HTML, mb))

    # A second build that needs no network at all.  Double-clicking the online
    # build on a plane gets you a blank white page, which is a poor way to
    # find out the dependency was there.
    if os.path.exists(THREE_LOCAL):
        print("wrote %s  (%.2f MB, no network needed)"
              % (OUT_OFFLINE, _write(OUT_OFFLINE, _offline(html))))
    else:
        print("  (no %s, skipping the offline build)" % THREE_LOCAL)
    m = data["meta"]
    print("  %s points, %s queries, %s lineage segments, %d families named"
          % (f"{data['n']:,}", f"{data['nq']:,}", f"{data['nl']:,}",
             len(m["families"])))
    if mb > 15.0:
        print("  WARNING: over the 16 MB artifact budget")


if __name__ == "__main__":
    main()
