const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const dir=(process.argv[2]||'results/experiments/m2_body_quality_v1').replace(/\/$/,'')+'/';
const report=JSON.parse(fs.readFileSync(dir+'report.json','utf8')),model=JSON.parse(fs.readFileSync(dir+'quality.json','utf8'));
const elements=new Map();function element(id){if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',hidden:false});return elements.get(id);}
const sandbox={window:{},$:element,show:(id,value=true)=>element(id).hidden=!value,esc:s=>String(s).replace(/</g,'&lt;'),api:async url=>url.includes('/draft/')?{ready:false,message:'尚未准备'}:{ready:true,trial_ready:report.eligible_for_research_trial,report,model,examples:[]}};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('src/web/static/body-quality-ui.js','utf8'),sandbox);
(async()=>{
 await sandbox.window.BodyQualityUI.load('screw');assert.equal(element('body-quality-form').hidden,!report.eligible_for_research_trial);
 assert.match(element('body-quality-metrics').innerHTML,/正常样本记忆库/);
 if(report.historical_regression){assert.match(element('body-quality-metrics').innerHTML,/历史998张回归检查/);assert.match(element('body-quality-metrics').innerHTML,/新的盲测/);assert.match(element('body-quality-metrics').innerHTML,/V1完整失败对照/);}
 else assert.match(element('body-quality-metrics').innerHTML,/有标签整图/);
 assert.doesNotMatch(element('body-quality-metrics').innerHTML,/undefined|NaN/);assert.match(element('body-quality-status').textContent,new RegExp(String(report.test_count)));
 await sandbox.window.BodyQualityUI.load('draft');assert.equal(element('body-quality-form').hidden,true);assert.equal(element('body-quality-metrics').innerHTML,'');
 console.log('Real BODY report rendered with candidate gate and draft isolation.');
})().catch(e=>{console.error(e);process.exitCode=1;});
