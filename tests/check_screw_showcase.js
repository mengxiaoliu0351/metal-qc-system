const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',handlers:{},querySelector:()=>({insertAdjacentElement:()=>{}}),addEventListener(type,fn){this.handlers[type]=fn;}});return elements.get(id);}
const buttons=['library','evaluation','workflow'].map(tab=>({dataset:{screwTab:tab},setAttribute(k,v){this[k]=v;}}));
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fixture={inventory:{organization:'grouped references',shared_encoder:'CLIP',router_version:'router1',appearance_version:'appearance1',appearance_reference_count:24,exclusion_reference_count:72,appearance_categories:['example'],view_groups:[{name:'fixture-view',quality_model_bound:false,status:'review only',reference_url:'/reference.png'}]},evidence_ready:true,
 evidence:{version:'fixture',built_utc:'fixture-time',datasets:[{name:'source',images:210,unit:'images',physical_specimens:70,quality_labels:true}],routing_matches_current:true,appearance_matches_current:true,routing:{known:120,known_accepted:115,unknown:65,unknown_accepted:0},appearance:{known:120,correct_accepted:112,unknown:184,unknown_accepted:0},quality_cohorts:[{name:'<unsafe>',population:'38 specimens',scope:'no VLM',state:'not released',matches_current_model:false,metrics:{accuracy:35/38,recall:16/19,false_positive_rate:0,false_positives:0,false_negatives:3,confusion_matrix:[[19,0],[3,16]]}}],limitations:['not a production claim']},operations:{new_policy_records:0,automation_rate:null,human_review_rate:null,vlm_call_rate:null,scope_note:'simulation excluded'},workflow:[{step:'01',title:'references',detail:'fixture'}],demo_note:'simulation only'};
let openedReview=null;
fixture.integration={checks:[{sample:'fixture',session_id:'123456_abcdef1234',quality_model_bound:false,vlm_called:false}],mixed_view_count:3,batch_id:'f'.repeat(32),completed_utc:'fixture-time',matches_current_workflow:true};
const sandbox={window:{ReviewUI:{openReference:x=>{openedReview=x;}}},document:{createElement:()=>element('screw-showcase'),querySelectorAll:()=>buttons},$:element,esc,api:async()=>structuredClone(fixture)};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('src/web/static/screw-showcase-ui.js','utf8'),sandbox);
(async()=>{
 const ui=sandbox.window.ScrewShowcaseUI;ui.init();await ui.load();
 assert.match(element('screw-showcase-content').innerHTML,/24/);assert.match(element('screw-showcase-content').innerHTML,/72/);
 const select=tab=>element('screw-showcase').handlers.click({target:{closest:selector=>selector==='[data-screw-tab]'?{dataset:{screwTab:tab}}:null}});
 select('evaluation');let html=element('screw-showcase-content').innerHTML;
 assert.match(html,/84\.2%/);assert.match(html,/not released/);assert.match(html,/0 \/ 3/);assert.match(html,/&lt;unsafe&gt;/);assert.doesNotMatch(html,/<unsafe>/);
 assert.match(html,/showcase\/export/);assert.equal(buttons[1]['aria-pressed'],'true');
 select('workflow');html=element('screw-showcase-content').innerHTML;assert.match(html,/暂无正式作业/);assert.match(html,/#family/);assert.doesNotMatch(html,/NaN%|undefined/);
 assert.match(html,/api\/report\/123456_abcdef1234/);
 element('screw-showcase').handlers.click({target:{closest:selector=>selector==='[data-screw-review]'?{dataset:{screwReview:'123456_abcdef1234'}}:null}});assert.equal(openedReview.session_id,'123456_abcdef1234');
 sandbox.api=async()=>{throw new Error('offline');};await ui.load();assert.equal(element('screw-showcase-content').innerHTML,'');assert.match(element('screw-showcase-status').textContent,/offline/);
 console.log('Screw evidence tabs, bounded metrics, failed candidates, escaping and unavailable-state behavior verified.');
})().catch(error=>{console.error(error);process.exitCode=1;});
