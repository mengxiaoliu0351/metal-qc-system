const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const directory=process.argv[2]||'results/experiments/screw_family_v1';
const report=JSON.parse(fs.readFileSync(path.join(directory,'report.json'),'utf8'));
const meta=JSON.parse(fs.readFileSync(path.join(directory,'family.json'),'utf8'));
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',hidden:false,querySelector:s=>element(id+s)});return elements.get(id);}
let selected='screw';
const sandbox={window:{FamilyCatalog:{load:async()=>{},gallery:()=>{},selected:()=>selected}},api:async()=>selected==='screw'?({ok:true,ready:true,report,model:meta}):({ok:true,ready:false,message:'尚无配套模型'}),$:element,show:(id,value=true)=>element(id).hidden=!value,esc:s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('src/web/static/family-ui.js','utf8'),sandbox);
(async()=>{
  await sandbox.window.FamilyUI.load();
  assert.match(element('family-metrics').innerHTML,/模拟变暗/);
  assert.match(element('family-memory').innerHTML,/正常校准图/);
  assert.match(element('family-details').innerHTML,/M2/);
  assert.equal(element('family-form').hidden,false);
  if(report.new_retained_evaluation){assert.match(element('family-fresh').innerHTML,/300张/);assert.match(element('family-fresh').innerHTML,/—/);assert.equal(element('family-prior').hidden,false);}
  const rendered=[...elements.values()].map(e=>e.innerHTML).join('');
  assert.doesNotMatch(rendered,/undefined|NaN/);
  selected='draft';await sandbox.window.FamilyUI.load(false);
  assert.equal(element('family-form').hidden,true);
  assert.equal(element('family-benchmark').hidden,true);
  assert.match(element('family-status').textContent,/尚无配套模型/);
  console.log('Actual family report rendered without missing fields: '+directory);
})().catch(e=>{console.error(e);process.exitCode=1;});
