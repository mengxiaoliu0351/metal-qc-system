// Execute the real catalogue renderer against the imported candidate manifest.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const manifest=JSON.parse(fs.readFileSync('results/product_families/screw/appearance.json','utf8'));
const elements=new Map();
function element(id){
  if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',value:'',hidden:false,disabled:false,
    listeners:{},addEventListener(event,callback){this.listeners[event]=callback;},appendChild(){},querySelectorAll(){return [];}});
  return elements.get(id);
}
const calls=[];
const sandbox={window:{},document:{createElement:()=>({})},$:element,esc:s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),api:async(url)=>{
  calls.push(url);
  if(url==='/api/product-families')return {families:[{key:'screw',name:'螺丝族',model_ready:true,appearance_summary:manifest.summary,reviewed_normal:0,eligible_normal_references:0},{key:'draft',name:'待建设族',model_ready:false,appearance_summary:{types:0,samples:0},reviewed_normal:0,eligible_normal_references:0}]};
  if(url.includes('/draft/'))return {summary:{samples:0,types:0},samples:[],types:[],count:0};
  return {...manifest,count:manifest.samples.length,samples:manifest.samples.slice(0,24).map(s=>({...s,quality_label:'unverified',appearance_status:'unreviewed',eligible_normal_reference:false,image_url:'/api/example/'+s.key}))};
}};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('src/web/static/family-catalog-ui.js','utf8'),sandbox);
(async()=>{
  let changes=0;sandbox.window.FamilyCatalog.init(()=>changes++);
  await sandbox.window.FamilyCatalog.load();await sandbox.window.FamilyCatalog.gallery();
  assert.match(element('family-gallery').innerHTML,/外观资料待审核/);assert.match(element('family-gallery').innerHTML,/质量标签：来源未提供/);
  assert.doesNotMatch(element('family-appearance').innerHTML,/批准为候选正常参考|所选标记正常|所选标记缺陷/);
  assert.match(element('family-appearance').innerHTML,/每类约5张/);
  assert.match(element('family-data-source').textContent,/MVTec Screws/);
  assert.doesNotMatch(element('family-gallery').innerHTML,/undefined|NaN/);
  assert.match(element('family-category').innerHTML,/type|螺丝|螺栓/);
  element('family-select').value='draft';await element('family-select').listeners.change();await sandbox.window.FamilyCatalog.gallery();
  assert.equal(changes,1);assert.equal(sandbox.window.FamilyCatalog.selected(),'draft');assert.equal(element('family-gallery').innerHTML,'');
  assert.match(element('family-gallery-status').textContent,/暂无候选/);
  console.log('Candidate catalogue rendered; draft selection clears prior images; no inferred quality labels.');
})().catch(e=>{console.error(e);process.exitCode=1;});
