// Runs the viewer's inline script against a stubbed THREE + DOM, so the
// decode path, the colouring and the UI wiring are exercised outside a
// browser.  WebGL itself is not tested; the shaders are checked by eye.
const fs = require('fs');
const path = require('path');

class V3 {
  constructor(x=0,y=0,z=0){ this.x=x; this.y=y; this.z=z; }
  set(x,y,z){ this.x=x; this.y=y; this.z=z; return this; }
  copy(v){ return this.set(v.x,v.y,v.z); }
  clone(){ return new V3(this.x,this.y,this.z); }
  lerpVectors(a,b,t){ return this.set(a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t); }
  addScaledVector(v,s){ this.x+=v.x*s; this.y+=v.y*s; this.z+=v.z*s; return this; }
  setFromMatrixColumn(m,i){ const e=m.elements; return this.set(e[i*4],e[i*4+1],e[i*4+2]); }
  setScalar(s){ return this.set(s,s,s); }
  length(){ return Math.hypot(this.x,this.y,this.z); }
}
class M4 {
  constructor(){ this.elements = new Float64Array(16); this.elements[0]=this.elements[5]=this.elements[10]=this.elements[15]=1; }
  multiplyMatrices(a,b){ return this; }
}
class Obj {
  constructor(){ this.children=[]; this.visible=true; this.position=new V3();
                 this.scale=new V3(1,1,1); this.matrix=new M4(); this.frustumCulled=true; }
  add(o){ this.children.push(o); return this; }
}
class BufAttr {
  constructor(array,itemSize,normalized){ this.array=array; this.itemSize=itemSize;
    this.normalized=!!normalized; this.needsUpdate=false; this.count=array.length/itemSize; }
}
class BufGeo {
  constructor(){ this.attributes={}; }
  setAttribute(n,a){ this.attributes[n]=a; return this; }
  setFromPoints(p){ this.pts=p; return this; }
}
const THREE = {
  NormalBlending: 1,
  Vector3: V3, Matrix4: M4, Group: Obj, Object3D: Obj,
  BufferGeometry: BufGeo, BufferAttribute: BufAttr,
  ShaderMaterial: class { constructor(o={}){ Object.assign(this,o);
    this.uniforms = o.uniforms || {}; if(!o.vertexShader||!o.fragmentShader)
      throw new Error('ShaderMaterial without shaders'); } },
  LineBasicMaterial: class { constructor(o={}){ Object.assign(this,o); } },
  SpriteMaterial: class { constructor(o={}){ Object.assign(this,o); } },
  CanvasTexture: class { constructor(c){ this.image=c; this.needsUpdate=false; } },
  Points: class extends Obj { constructor(g,m){ super(); this.geometry=g; this.material=m; } },
  LineSegments: class extends Obj { constructor(g,m){ super(); this.geometry=g; this.material=m; } },
  Line: class extends Obj { constructor(g,m){ super(); this.geometry=g; this.material=m; } },
  Sprite: class extends Obj { constructor(m){ super(); this.material=m; } },
  Scene: class extends Obj {},
  PerspectiveCamera: class extends Obj {
    constructor(fov,a,n,f){ super(); this.fov=fov; this.aspect=a;
      this.projectionMatrix=new M4(); this.matrixWorldInverse=new M4(); }
    updateProjectionMatrix(){} lookAt(){}
  },
  WebGLRenderer: class {
    constructor(){ this.domElement = makeEl('canvas'); }
    setPixelRatio(){} setClearColor(){} setSize(){}
    render(scene){ this.rendered=(this.rendered||0)+1; lastScene = scene; }
  },
};
let lastScene = null;         // what the page last drew, for reading uniforms back

// ---------------------------------------------------------------- fake DOM --
const listeners = [];
function makeEl(tag){
  const el = {
    tagName: tag, style:{}, children:[], value:'0', textContent:'', innerHTML:'',
    className:'', dataset:{}, checked:false,
    classList:{ _s:new Set(),
      add(c){this._s.add(c);}, remove(c){this._s.delete(c);},
      toggle(c,on){ on===undefined ? (this._s.has(c)?this._s.delete(c):this._s.add(c))
                                   : (on?this._s.add(c):this._s.delete(c)); },
      contains(c){return this._s.has(c);} },
    appendChild(c){ this.children.push(c); return c; },
    setAttribute(k,v){ (this.attrs = this.attrs || {})[k] = String(v); },
    remove(){}, setPointerCapture(){}, releasePointerCapture(){},
    addEventListener(t,f,o){ listeners.push([tag,t,f]); },
    getBoundingClientRect(){ return {left:0,top:0,width:1280,height:800}; },
    getContext(){ return ctx2d(); },
  };
  Object.defineProperty(el,'firstChild',{get(){return this.children[0]||null;}});
  return el;
}
function ctx2d(){
  const noop = ()=>{};
  return new Proxy({}, {get:(t,k)=> (k==='canvas' ? {width:128,height:128} : noop)});
}
const byId = {};
const document = {
  createElement: makeEl,
  getElementById(id){ return byId[id] || (byId[id] = makeEl('div')); },
  addEventListener(t,f){ listeners.push(['document',t,f]); },
};
global.document = document;
global.window = global;
global.innerWidth = 1280; global.innerHeight = 800; global.devicePixelRatio = 2;
global.addEventListener = (t,f)=>listeners.push(['window',t,f]);
global.THREE = THREE;
global.performance = { now: ()=>Date.now() };
global.matchMedia = (q)=>({matches:false, media:q, addEventListener(){}, addListener(){}});
let rafCount = 0, rafCb = null;
global.requestAnimationFrame = (f)=>{ rafCb = f; return ++rafCount; };
global.setTimeout = (f,ms)=>{ return 0; };            // do not defer during the check
global.devicePixelRatio = 2;
// `node _harness.js lake3d.html #before` opens the page the way that link does
const hashes = [];
if (process.argv[3]) global.location = {hash: process.argv[3], pathname: '/', search: ''};
global.history = {replaceState(_s, _t, url){ hashes.push(url); }};

// ------------------------------------------------------------------- run it --
const html = fs.readFileSync(path.join(__dirname, process.argv[2] || 'lake3d.html'), 'utf8');
const all = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/g)];
const m = all[all.length-1];          // the offline build inlines three.js first
if (!m) { console.error('no inline script found'); process.exit(1); }
const t0 = Date.now();
try {
  (0, eval)(m[1]);
} catch (e) {
  console.error('THREW:', e && e.stack || e);
  process.exit(1);
}
console.log('script ran clean in', Date.now()-t0, 'ms');

// drive a few frames through the animation loop
for (let i = 0; i < 3 && rafCb; i++){ const f = rafCb; rafCb = null; f(performance.now()); }
console.log('frames driven:', 3);

// poke the UI the way a user would
const modes = byId['modes'];
console.log('mode chips:', modes.children.length,
            modes.children.map(c=>c.textContent).join(','));
modes.children.forEach(c=>c.onclick && c.onclick());
const fams = byId['fams'];
console.log('family rows:', fams.children.length);
if (fams.children[0] && fams.children[0].onclick){
  fams.children[0].onclick();                        // isolate
  fams.children[0].onclick();                        // release
}
['B_reset','B_top','B_cast','B_fly'].forEach(id=>{
  const b = byId[id]; if (b && b.onclick) b.onclick({target:b});
});
['S_size','S_op','S_spin'].forEach(id=>{
  const s = byId[id]; if (s && s.oninput){ s.value = '2.5'; s.oninput(); }
});
['L_models','L_queries','L_lineage'].forEach(id=>{
  const c = byId[id]; if (c && c.onchange) c.onchange({target:{checked:true}});
});
for (let i = 0; i < 3 && rafCb; i++){ const f = rafCb; rafCb = null; f(performance.now()); }

// the two spaces: every material shares one mix uniform, the switch moves it
// to the far end once the tween's time has passed, and the page says which
// space it is showing in the masthead, the panel and the address
function drive(later){
  for (let i = 0; i < 3 && rafCb; i++){ const f = rafCb; rafCb = null; f(performance.now() + later); }
}
function mixes(){
  const out = new Set();
  (function walk(o){
    if (o.material && o.material.uniforms && o.material.uniforms.uMix)
      out.add(o.material.uniforms.uMix);
    (o.children || []).forEach(walk);
  })(lastScene);
  return [...out];
}
const spaces = byId['spaces'];
console.log('space chips:', spaces.children.map(c=>c.textContent).join(' | '),
            ' on:', spaces.children.map(c=>c.classList.contains('on') ? 1 : 0).join(''));
const shared = mixes();
if (shared.length !== 1) { console.error('materials do not share one mix uniform:', shared.length); process.exit(1); }
const start = shared[0].value;
if (start !== (process.argv[3] === '#before' ? 0 : 1)) { console.error('opened on mix', start); process.exit(1); }
if (hashes.length) { console.error('opening the page rewrote the address:', hashes); process.exit(1); }
const target = start ? 0 : 1;
spaces.children[target].onclick();
drive(0);
const mid = shared[0].value;
drive(60000);
console.log('switch', start, '->', target, ': mix', start, '->', mid.toFixed(3), '->', shared[0].value,
            ' on:', spaces.children.map(c=>c.classList.contains('on') ? 1 : 0).join(''),
            ' address:', hashes[hashes.length-1]);
if (shared[0].value !== target) { console.error('the switch did not arrive'); process.exit(1); }
console.log('masthead:', byId['sub'].textContent);
console.log('panel   :', byId['spacenote'].innerHTML.replace(/<\/tr>/g,' | ').replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').slice(0,330));
console.log('cast    :', byId['castinfo'].innerHTML.replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').slice(0,300));
byId['B_fly'].onclick({target:byId['B_fly']});
spaces.children[start].onclick();
drive(60000);
if (shared[0].value !== start) { console.error('the switch did not come back'); process.exit(1); }

console.log('masthead:', byId['sub'].textContent);
console.log('foot    :', byId['foot'].innerHTML.replace(/<br>/g,' | '));
console.log('cast    :', byId['castinfo'].innerHTML.slice(0,120).replace(/<[^>]+>/g,''));
console.log('listeners registered:', listeners.length);
console.log('OK');
