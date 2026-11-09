// Execute the actual frontend in an isolated DOM. No browser, network, labels or model writes.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),vm=require('node:vm');
const {JSDOM,VirtualConsole}=require(process.env.METALQC_DOM_MODULE||'jsdom');
const root=path.resolve(__dirname,'..'),assets=path.join(root,'src/web/static');
const errors=[],requests=[];const virtualConsole=new VirtualConsole();virtualConsole.on('jsdomError',e=>errors.push(e.message));
const dom=new JSDOM(fs.readFileSync(path.join(assets,'index.html'),'utf8'),{url:'http://localhost/#detect',runScripts:'outside-only',pretendToBeVisual:true,virtualConsole});
const w=dom.window;w.scrollTo=()=>{};w.HTMLElement.prototype.scrollIntoView=()=>{};w.URL.createObjectURL=()=> 'blob:fixture';w.URL.revokeObjectURL=()=>{};
const products=[{key:'metal_nut',name:'螺母',threshold:.29,t_high:.65,method:'test'},{key:'screw',name:'螺丝近景',threshold:.34,t_high:.75,method:'test'},{key:'head',name:'头部配置',threshold:.38,t_high:.68,method:'test'}];
const targets=[{key:'metal_nut',name:'螺母',mode:'product',product_key:'metal_nut',batch_key:'metal_nut',ready:true},{key:'screw',name:'螺丝',mode:'family',family_key:'screw',batch_key:'family:screw',ready:true}];
const responses={
 '/api/products':{products,inspection_targets:targets},
 '/api/product-families/screw/inspection':{ready:true,examples:[]},
 '/api/inspection/policy':{mode:'assisted',sample_rate:.1,audit_period:'day',holds:{'head:current-head':{reason:'当前版本仅辅助评分'},'metal_nut:old-nut':{reason:'old version'}},scope:'fixture'},
 '/api/review/queue':{records:[],count:0,summary:{pending:2,sampled:1,reviewed:4,false_positive:1,false_negative:1}},
 '/api/iterations':{runs:[],pools:{}},'/api/history':{records:[]},'/api/product-catalog':{products:[{key:'head',name:'头部配置',view:'俯视',sku:'M2',source:'public fixture',license:'fixture',active_build:'active',counts:Object.fromEntries(['bank','train','calibration','qualification','test','feedback'].map(k=>[k,{normal:10,defect:k==='bank'?0:10}])),builds:[{id:'active',status:'completed',message:'fixture complete'}]}]},
 '/api/stats':{operations:{new_policy_records:10,automation_rate:.6,pending_review:4,vlm_call_rate:.2,median_seconds:1.5,automatic_normal:4,automatic_isolated:2,sent_to_review:4,reviewed_records:0,labelled_comparisons:0,reviewed_agreement:null,false_positive:0,false_negative:0},total:10},
 '/api/inspection/evidence':{profiles:[{product:'head',model_key:'current-head',allow_normal:true,allow_isolate:true},{product:'metal_nut',model_key:'current-nut',allow_normal:true,allow_isolate:true}]},
 '/api/inspection/audit':{period_key:'fixture-day',audit_period:'day',automatic:6,selected:2,selected_pending:1,confirmed_errors:0},'/api/inspection/gray-evidence':{total:2,auto_normal:0,auto_defect:0,uncertain:2},
 '/api/product-families':{families:[{key:'screw',name:'螺丝族',model_ready:true,appearance_summary:{types:3,samples:6}}]},
 '/api/product-family/benchmark':{ready:false,message:'fixture research unavailable'},
 '/api/product-families/screw/appearance':{samples:[],types:[],summary:{samples:0,types:0},count:0},
 '/api/batches':{jobs:[]}
};
w.fetch=async (url,options={})=>{requests.push({url,method:options.method||'GET',body:options.body});if(url==='/api/inspection/policy'&&options.method==='POST')Object.assign(responses[url],JSON.parse(options.body));return {ok:true,status:200,json:async()=>responses[String(url).split('?')[0]]||{ok:true,records:[],products:[],runs:[]}};};
const execute=code=>vm.runInContext(code,dom.getInternalVMContext());
for(const node of w.document.querySelectorAll('script[src]')){const file=node.getAttribute('src').split('/').pop().split('?')[0];execute(fs.readFileSync(path.join(assets,file),'utf8'));}
const tick=()=>new Promise(resolve=>setTimeout(resolve,40));
(async()=>{
 await tick();assert.deepEqual(errors,[],'app must bootstrap without DOM errors');
 const doc=w.document;
 assert.equal(doc.querySelectorAll('[id]').length,new Set([...doc.querySelectorAll('[id]')].map(n=>n.id)).size,'duplicate IDs break labels and actions');
 assert.deepEqual([...doc.querySelectorAll('.daily-navigation [data-page]')].map(n=>n.dataset.page),['detect','batch','review','history']);
 assert.equal(doc.querySelector('#nav-advanced').open,false);
 assert.equal(doc.querySelectorAll('.chat-panel').length,1);assert.equal(doc.querySelectorAll('[data-station-tab]').length,0);
 assert.equal(doc.querySelector('.score-details').open,false);
 assert.equal(doc.querySelector('.score-details').closest('#analysis-content').id,'analysis-content','score is secondary evidence, not a competing first outcome');
 doc.querySelector('.workspace-assistant').click();await tick();assert.equal(doc.querySelector('#page-assistant').hidden,false);assert.equal(doc.querySelector('#page-detect').hidden,true);
 doc.querySelector('#assistant-back').click();await tick();assert.equal(doc.querySelector('#page-detect').hidden,false);
 doc.querySelector('[data-page="iterations"]').click();await tick();assert.equal(doc.querySelector('#nav-advanced').open,true);assert.equal(doc.querySelector('#page-iterations').hidden,false);
 doc.querySelector('[data-page="review"]').click();await tick();assert.equal(doc.querySelector('#nav-advanced').open,false);assert.equal(doc.querySelectorAll('#review-summary>div').length,4);
 assert.match(doc.querySelector('#review-summary').textContent,/已核验不一致2/);
 // Render both outcomes with the real renderer: exactly one primary next action.
 const base={ok:true,product:'螺丝',score:.1,threshold:.34,t_high:.75,route:'low_risk',session_id:'fixture',heatmap:'aGVsbG8=',cognition:{ok:true,conclusion:'系统判正常',defect_desc:'fixture',suggestion:'fixture'}};
 execute('goPage("detect")');w.__fixture=base;
 execute('renderResult({...window.__fixture,disposition:{requires_review:false,label:"系统正常归档",reason:"fixture",status:"auto_normal"}},1)');
 assert.equal(doc.querySelector('#review-button').hidden,true);assert.equal(doc.querySelector('#next-file').hidden,false);
 execute('renderResult({...window.__fixture,disposition:{requires_review:true,label:"待人工核验",reason:"fixture",status:"needs_review"}},1)');
 assert.equal(doc.querySelector('#review-button').hidden,false);assert.equal(doc.querySelector('#next-file').hidden,true);
 assert.equal(doc.querySelector('#report-button').hidden,false);
 doc.querySelector('#explain-button').click();await tick();assert.equal(doc.querySelector('#page-assistant').hidden,false);assert.match(doc.querySelector('#chat-context').textContent,/已附带当前检测/);
 doc.querySelector('#assistant-back').click();await tick();assert.equal(doc.querySelector('#analysis-content').hidden,false,'return must preserve the detection result');
 doc.querySelector('#product-select').value='screw';doc.querySelector('#product-select').dispatchEvent(new w.Event('change'));
 const file=new w.File(['x'],'part.png',{type:'image/png'});w.__file=file;execute('chooseFile(window.__file)');
 assert.equal(doc.querySelector('#detect-button').disabled,false);assert.match(doc.querySelector('#file-name').textContent,/part.png/);
 doc.querySelector('#clear-file').click();assert.equal(doc.querySelector('#detect-button').disabled,true);
 // Management pages have distinct responsibilities; maintenance data is lazy.
 let checkpoint=requests.length;doc.querySelector('[data-page="products"]').click();await tick();
 assert.equal(doc.querySelector('#settings-panel-config').hidden,false);assert.equal(doc.querySelector('#settings-panel-samples').hidden,true);assert.equal(doc.querySelector('#settings-panel-research').hidden,true);
 assert.ok(!requests.slice(checkpoint).some(r=>/product-family\/benchmark|\/matching|body-quality|\/appearance/.test(r.url)),'config must not fetch research reports or images');
 assert.match(doc.querySelector('#runtime-config-list').textContent,/头部配置test已暂停，全部复核/,'current model hold overrides an eligible historical profile');
 assert.match(doc.querySelector('#runtime-config-list').textContent,/螺母test按门槛自动处理/,'a hold on another model version must not block this one');
 assert.equal(doc.querySelectorAll('#policy-form').length,1);assert.equal(doc.querySelector('#policy-form').closest('#settings-panel-config').id,'settings-panel-config');
 doc.querySelector('#policy-rate').value='.2';doc.querySelector('#policy-rate').value='0.2';doc.querySelector('#policy-rate').dispatchEvent(new w.Event('change',{bubbles:true}));
 doc.querySelector('[data-settings-tab="samples"]').click();await tick();assert.equal(doc.querySelector('#catalog-create-panel').open,false);assert.match(doc.querySelector('#catalog-next-step').textContent,/已接入检测/);
 assert.equal(doc.querySelectorAll('#catalog-builds [data-catalog-batch]').length,2,'sealed evaluation and replay tasks stay available');
 doc.querySelector('[data-settings-tab="config"]').click();await tick();assert.equal(doc.querySelector('#policy-rate').value,'0.2','unsaved form changes survive tab switches and read refresh');
 doc.querySelector('[data-settings-tab="research"]').click();await tick();assert.equal(doc.querySelector('#research-appearance').hidden,true);assert.equal(doc.querySelector('#research-trial').hidden,true);
 assert.ok(!requests.some(r=>/\/matching|body-quality/.test(r.url)),'advanced trials must not load just by opening reports');
 doc.querySelector('[data-research-view="appearance"]').click();await tick();assert.equal(doc.querySelector('#research-appearance').hidden,false);assert.ok(requests.some(r=>/screw\/appearance/.test(r.url)));
 doc.querySelector('[data-page="dashboard"]').click();await tick();assert.equal(doc.querySelectorAll('#page-dashboard #policy-form').length,0);assert.equal(doc.querySelector('#automation-rate').textContent,'60.0%');assert.equal(doc.querySelector('#pending-tasks').textContent,'4');assert.match(doc.querySelector('#quality-reviewed').textContent,/—/);
 assert.equal(doc.querySelector('#dashboard-details').open,false);assert.ok(!requests.some(r=>r.url==='/api/inspection/gray-evidence'),'gray detail is loaded only on expansion');
 doc.querySelector('#dashboard-sampled').click();await tick();assert.equal(doc.querySelector('#review-filter').value,'sampled');
 doc.querySelector('[data-page="dashboard"]').click();await tick();checkpoint=requests.length;doc.querySelector('#dashboard-policy').click();await tick();assert.equal(doc.querySelector('#settings-panel-config').hidden,false);assert.ok(!requests.slice(checkpoint).some(r=>/product-family\/benchmark/.test(r.url)),'policy shortcut must not load the previously selected research tab');
 assert.deepEqual(errors,[]);assert.equal(requests.filter(r=>r.method!=='GET').length,0,'all viewing/navigation must be read-only');
 doc.querySelector('#policy-operator').value='  Operator  ';doc.querySelector('#policy-period').value='week';
 execute('state.busy=true');doc.querySelector('#policy-form').dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));await tick();assert.equal(requests.filter(r=>r.method==='POST').length,0,'active detection blocks policy edits');
 execute('state.busy=false');doc.querySelector('#policy-form').dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));await tick();
 const writes=requests.filter(r=>r.method!=='GET');assert.equal(writes.length,1);assert.equal(writes[0].url,'/api/inspection/policy');assert.deepEqual(JSON.parse(writes[0].body),{mode:'assisted',sample_rate:.2,audit_period:'week',operator:'Operator'});
 assert.match(doc.querySelector('#policy-message').textContent,/已保存/);assert.ok(!/undefined|NaN/.test(doc.querySelector('#page-dashboard').textContent));assert.deepEqual(errors,[]);
 console.log('Actual DOM journeys passed: inspection/assistant, dashboard queue shortcuts, three settings tabs, model-specific holds, lazy research, retained sample/build actions, dirty form preservation, read-only navigation and explicit busy-guarded policy save.');
 dom.window.close();
})().catch(e=>{dom.window.close();console.error(e);process.exitCode=1;});
