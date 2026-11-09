const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=(process.argv[2]||'results/experiments/screw_appearance_v1').replace(/\/$/,'')+'/';
const report=JSON.parse(fs.readFileSync(root+'report.json','utf8')),model=JSON.parse(fs.readFileSync(root+'appearance.json','utf8'));
const elements=new Map();function element(id){if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',hidden:false});return elements.get(id);}
const sandbox={window:{},$:element,show:(id,value=true)=>element(id).hidden=!value,esc:s=>String(s).replace(/</g,'&lt;'),api:async url=>url.includes('/draft/')?{ready:false,message:'正在准备'}:{ready:true,report,model}};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('src/web/static/appearance-ui.js','utf8'),sandbox);
(async()=>{
 await sandbox.window.AppearanceUI.load('screw');assert.equal(element('appearance-form').hidden,false);
 const m=report.methods[report.selected_method].test;
 assert.ok(element('appearance-metrics').innerHTML.includes(`${m.accepted_other} 张被误匹配`));
 assert.ok(element('appearance-metrics').innerHTML.includes((100*m.accepted_precision_all).toFixed(1)+'%'));
 assert.match(element('appearance-status').textContent,/24 张/);
 if(report.comparisons){assert.match(element('appearance-status').textContent,/72 张来源类别排除参考/);assert.match(element('appearance-metrics').innerHTML,/开发回归/);assert.ok(element('appearance-metrics').innerHTML.includes(report.version));}
 assert.doesNotMatch(element('appearance-metrics').innerHTML,/undefined|NaN/);
 await sandbox.window.AppearanceUI.load('draft');assert.equal(element('appearance-form').hidden,true);assert.equal(element('appearance-benchmark').hidden,true);
 console.log('Real appearance report rendered with unknown false accepts and draft isolation.');
})().catch(e=>{console.error(e);process.exitCode=1;});
