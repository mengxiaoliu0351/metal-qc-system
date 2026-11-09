'use strict';
window.ReviewUI = (()=>{
  let selection=null, lastReviewer='', offset=0, poll, queueRequest=0, runRequest=0;
  const percent=n=>Number.isFinite(n)?(n*100).toFixed(1)+'%':'—';
  const truthLabel=v=>v==='ok'?'正常 / 合格':v==='defect'?'存在缺陷':'待核验';
  const prediction=v=>v==='normal'?'正常':v==='defect'?'缺陷':'未自动判定';
  const html=`
  <section id="page-review" class="page" hidden aria-label="人工复核">
    <div class="page-heading"><div class="heading-copy"><h1>人工最终复核</h1><p>先处理待办，核对图像后保存最终标签；自动记录可在检验档案中抽查。</p></div><button id="review-refresh" class="button secondary">刷新复核队列</button></div>
    <div id="review-error" class="inline-error" role="alert" hidden></div>
    <div id="review-summary" class="review-summary"></div>
    <div class="panel" id="review-queue-panel"><div class="review-toolbar"><label>产品<select id="review-product"><option value="">全部产品</option></select></label><label>复核状态<select id="review-filter"><option value="pending">待处理任务</option><option value="gray">灰区待核验</option><option value="sampled">抽查任务</option><option value="automatic">自动已处理</option><option value="legacy">历史未分流</option><option value="all">全部记录</option><option value="reviewed">已复核</option><option value="errors">系统判定有误</option><option value="flagged">反馈标记待复核</option></select></label><span id="review-count" class="caption"></span></div><div class="table-scroll"><table><thead><tr><th>检验时间 / 样品</th><th>产品</th><th>系统原始判定</th><th>人工真实标签</th><th>任务来源 / 一致性</th><th></th></tr></thead><tbody id="review-table"></tbody></table></div><div class="review-pagination"><button id="review-prev" class="button secondary">上一页</button><span id="review-page" class="caption"></span><button id="review-next" class="button secondary">下一页</button></div></div>
    <section id="review-detail" class="panel review-detail" hidden aria-live="polite"></section>
  </section>
  <section id="page-iterations" class="page" hidden aria-label="模型与阈值迭代">
    <div class="page-heading"><div class="heading-copy"><h1>模型与阈值迭代</h1><p>人工复核 → 去重分区 → 候选训练/校准 → 回归评估 → 发布 / 回滚。</p></div><button id="iteration-refresh" class="button secondary">刷新任务</button></div>
    <section class="panel"><form id="iteration-form" class="iteration-controls"><label>产品<select id="iteration-product" required></select></label><label>迭代内容<select id="iteration-mode"><option value="model">重训监督判别头 + 校准双阈值</option><option value="threshold">保持模型，仅校准双阈值</option></select></label><button id="iteration-start" class="button primary" type="submit">生成候选并评估</button></form><p id="iteration-pool" class="panel-footnote"></p><details class="iteration-guide"><summary>样本使用与发布规则</summary><p class="panel-footnote">复核时勾选“批准用于迭代”的样本才会参与。相同图像自动去重；冻结验证/测试图片不会回流训练。新增图像按内容哈希固定分为训练、校准和审计组。固定 CLIP 主干与正常记忆库，仅更新监督判别头和门控阈值。</p></details></section>
    <div id="iteration-error" class="inline-error" role="alert" hidden></div><div id="iteration-runs" class="iteration-runs"></div>
  </section>`;
  function setProducts(products){
    const previous=$('review-product').value, active=$('iteration-product').value;
    const options=products.map(p=>`<option value="${esc(p.key)}">${esc(p.name)}</option>`).join('');
    $('review-product').innerHTML='<option value="">全部产品</option>'+options;
    $('iteration-product').innerHTML=options;
    if(products.some(p=>p.key===previous))$('review-product').value=previous;
    if(products.some(p=>p.key===active))$('iteration-product').value=active;
    if(!$('page-iterations').hidden)loadRuns();
  }
  function failure(id,error){$(id).textContent=error.message||String(error);show(id);}
  async function loadQueue(){
    const request=++queueRequest;show('review-error',false);$('review-refresh').disabled=true;
    try{
      const data=await api('/api/review/queue?'+new URLSearchParams({product:$('review-product').value,status:$('review-filter').value,offset,limit:50}));
      if(request!==queueRequest)return;
      const s=data.summary;
      $('review-summary').innerHTML=[['待处理任务',s.pending],['其中抽查',s.sampled],['已完成核验',s.reviewed],['已核验不一致',(s.false_positive??0)+(s.false_negative??0)]].map(([label,n])=>`<div><span>${label}</span><strong class="mono">${n}</strong></div>`).join('');
      $('review-count').textContent=`匹配 ${data.count} 条；一致性仅统计已复核样本，不等于独立测试精度`;
      $('review-table').innerHTML=data.records.length?data.records.map(r=>`<tr><td><span class="mono">${esc((r.timestamp||'').slice(0,19).replace('T',' '))}</span><small>${esc(r.sample_name||'未命名')}</small></td><td>${esc(r.product_name||r.product_key||'未记录')}</td><td>${prediction(r.system_prediction)}<small class="mono">${fmt(r.score)}</small></td><td>${truthLabel(r.human_verdict?.verdict)}</td><td><small>${esc(r.disposition?.label||'历史未分流')}</small>${r.agreement===true?'<span class="badge low">一致</span>':r.agreement===false?'<span class="badge high">不一致</span>':'<span class="badge neutral">待核验</span>'}</td><td><button class="text-button" data-review-day="${esc(r._day)}" data-review-key="${esc(r._file)}">打开复核 →</button></td></tr>`).join(''):'<tr><td colspan="6">当前筛选下没有待处理的记录。</td></tr>';
      $('review-page').textContent=data.count?`${offset+1}–${Math.min(offset+50,data.count)} / ${data.count}`:'0 / 0';
      $('review-prev').disabled=offset===0;$('review-next').disabled=offset+50>=data.count;
    }catch(e){failure('review-error',e);}finally{if(request===queueRequest)$('review-refresh').disabled=false;}
  }
  function openQueue(status='pending'){offset=0;$('review-filter').value=status;goPage('review');}
  function showQueue(){
    show('review-summary');show('review-queue-panel');show('review-refresh');show('review-detail',false);selection=null;++detailToken;
  }
  async function openReference(reference){
    goPage('review');
    try{
      const id=reference.day?reference:await api('/api/review/record?'+new URLSearchParams({session_id:reference.session_id}));
      await openDetail(id.day,id.key);
    }catch(e){failure('review-error',e);}
  }
  let detailToken=0;
  async function openDetail(day,key){
    const token=++detailToken;selection=null;show('review-detail');$('review-detail').innerHTML='<span class="spinner"></span> 正在读取复核证据…';
    try{
      const data=await api('/api/history/detail?'+new URLSearchParams({day,key}));if(token!==detailToken)return;
      const r=data.record, a=r.human_verdict||{};
      selection={day,key,record:r,opened:performance.now()};
      const system=r.disposition?.auto_result||(r.gate?.route==='vlm_review'&&r.disposition?null:r.perception_prediction||(Number.isFinite(r.score)&&Number.isFinite(r.gate?.threshold)?(r.score>=r.gate.threshold?'defect':'normal'):null));
      $('review-detail').innerHTML=`<div class="review-detail-toolbar"><button class="button secondary" id="review-back" type="button">← 返回复核列表</button><span>核对图像 → 确认真实标签 → 保存复核</span></div><div class="panel-heading"><h2>${esc(r.sample_name||'工件复核')}</h2><a class="button secondary" href="/api/history/report?${new URLSearchParams({day,key})}" target="_blank" rel="noopener">导出 PDF 报告</a></div><div class="review-evidence"><div class="review-images">${[['input_img','原始图像'],['heatmap_img','局部异常热力图']].map(([field,label])=>`<figure><figcaption>${label}</figcaption>${data[field]?`<img src="${imageSrc(data[field])}" alt="${label}">`:'<p>图像缺失</p>'}</figure>`).join('')}<p class="panel-footnote">系统原始判定：<strong>${prediction(system)}</strong> · 评分 ${fmt(r.score)} · 下阈值 ${fmt(r.gate?.threshold)}<br>AI 建议：${esc(r.cognition?.conclusion||'未记录')}。一致性按系统自动归档结论对照人工标签；仍不确定的灰区不计作正确。</p><p class="panel-footnote"><strong>${esc(r.current_disposition?.label||'主动核验历史记录')}</strong> · ${esc(r.current_disposition?.reason||'')}</p><p class="detail-analysis">${esc(r.cognition?.defect_desc||'')}</p></div><form id="review-form" class="review-form"><h3>检验人员最终核验</h3><label>真实标签<select id="review-verdict" required><option value="">请核对图像后选择</option><option value="ok" ${a.verdict==='ok'?'selected':''}>正常 / 合格</option><option value="defect" ${a.verdict==='defect'?'selected':''}>存在缺陷 / 不合格</option></select></label><label>缺陷类型<input id="review-defect" maxlength="100" placeholder="例如：划痕、弯曲、裂纹；未知可留空" value="${esc(a.defect_type||'')}"></label><label>复核人员<input id="reviewer-name" required maxlength="80" placeholder="姓名或操作员工号" value="${esc(a.reviewer||lastReviewer)}"></label><label>核验依据与备注<textarea id="review-note" rows="3" maxlength="2000" placeholder="记录误报/漏检原因、可见缺陷位置或核验依据">${esc(a.note||'')}</textarea></label><label class="review-consent"><input type="checkbox" id="review-approve" ${a.confirmed&&a.use_for_iteration?'checked':''}><span>批准该最终标签用于离线迭代评估</span></label><p class="panel-footnote">提交会保留原始判定和复核修订记录。仅训练/校准候选版本，发布前仍需通过对比评估。</p><button id="review-submit" class="button primary" type="submit">提交最终复核</button><p id="review-saved" role="status">${a.confirmed?`已由 ${esc(a.reviewer||'操作员')} 于 ${esc(a.ts||'')} 复核；${a.agreement===true?'系统判定一致':a.agreement===false?'系统判定有误':'一致性未记录'}。`:''}</p></form></div>`;
      if(r.iteration_eligible===false){$('review-approve').disabled=true;$('review-approve').checked=false;$('review-approve').parentElement.querySelector('span').textContent='尚未绑定经验证的质量模型：本次仅保存最终核验，不回流已有模型';}
      if(r.disposition?.gray_assessment){const g=r.disposition.gray_assessment;$('review-detail').querySelector('.review-images').insertAdjacentHTML('beforeend',`<p class="panel-footnote">灰区深度分析：<strong>${g.proposal==='normal'?'系统建议正常':g.proposal==='defect'?'系统建议疑似缺陷':'仍不确定'}</strong> · ${esc(g.reason)}。${r.disposition.auto_result?'已按作业规则自动记录，抽查仍须核验。':'已转人工复核。'}</p>`);}
      if(r.record_scope==='simulation'&&r.family_key){$('review-detail').querySelector('.review-images').insertAdjacentHTML('beforeend','<p class="panel-footnote">产品族固定流程演示：用于检查匹配、检测和复核是否接通，不提供人工质量真值，不计入日常质效。</p>');}
      else if(r.record_scope==='simulation'){const s=r.source_annotation||{};$('review-detail').querySelector('.review-images').insertAdjacentHTML('beforeend',`<p class="panel-footnote">公开标签核验演示 · 来源标签：<strong>${s.label==='normal'?'正常':s.label==='defect'?'缺陷':'请核对来源'}</strong> · ${esc(s.source||'')}。本次核验以数据集标签为依据，记录不计入日常作业质量。</p>`);}
      show('review-queue-panel',false);show('review-summary',false);show('review-refresh',false);
      $('review-back').addEventListener('click',()=>{showQueue();window.scrollTo(0,0);});
      $('review-form').addEventListener('submit',submitReview);
      $('review-detail').scrollIntoView({behavior:'smooth',block:'start'});
    }catch(e){if(token===detailToken)$('review-detail').innerHTML=`<div class="inline-error">${esc(e.message)}</div>`;}
  }
  async function submitReview(event){
    event.preventDefault();if(!selection)return;
    const current=selection;
    $('review-submit').disabled=true;
    const request={day:current.day,key:current.key,session_id:current.record.session_id,verdict:$('review-verdict').value,defect_type:$('review-defect').value,reviewer:$('reviewer-name').value.trim(),note:$('review-note').value,review_seconds:Math.min(7200,(performance.now()-current.opened)/1000),use_for_iteration:$('review-approve').checked};
    try{
      const data=await api('/api/review/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
      lastReviewer=request.reviewer;
      if(selection===current){$('review-saved').textContent=`最终标签已保存。${data.annotation.agreement===true?'系统判定一致。':data.annotation.agreement===false?'系统判定有误，已记录。':''}${request.use_for_iteration?'已批准用于迭代评估。':''}`;current.record.human_verdict=data.annotation;}
      await loadQueue();toast(data.note);if(data.automation_suspended){$('review-saved').textContent+=' 当前产品版本自动处置已暂停，请到模型迭代评估修正。';}
    }catch(e){failure('review-error',e);}finally{if(selection===current&&$('review-submit'))$('review-submit').disabled=false;}
  }
  function metricsTable(report){
    const rows=[];
    for(const [key,label] of [['validation','校准集'],['regression','冻结回归集'],['new_audit','新增审计组']]){
      if(!report[key])continue;
      for(const [mode,text] of [['current','当前'],['candidate','候选']]){
        const m=report[key][mode];
        rows.push(`<tr><td>${label}</td><td>${text}</td><td>${m.n}</td><td>${percent(m.accuracy)}</td><td>${percent(m.recall)}</td><td>${m.confusion_matrix[0][1]}</td><td>${m.confusion_matrix[1][0]}</td><td>${percent(m.gray_rate)}</td></tr>`);
      }
    }
    return `<div class="table-scroll"><table><thead><tr><th>分区</th><th>版本</th><th>数量</th><th>准确率</th><th>缺陷召回</th><th>误报</th><th>漏检</th><th>灰区占比</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  }
  async function loadRuns(){
    const request=++runRequest;clearTimeout(poll);show('iteration-error',false);
    try{
      const data=await api('/api/iterations');
      if(request!==runRequest)return;
      const product=$('iteration-product').value;
      if(product){const summary=await api('/api/review/queue?'+new URLSearchParams({product,status:'reviewed',limit:1}));$('iteration-pool').textContent=`该产品已复核 ${summary.summary.reviewed} 件，批准用于迭代 ${summary.summary.iteration_approved} 件；当前版本：${data.active_versions[product]||'初始模型'}。`;}
      if(request!==runRequest)return;
      const runs=data.runs.filter(r=>!product||r.product===product);
      const running=data.runs.some(r=>r.status==='queued'||r.status==='running');
      $('iteration-start').disabled=running||!product||Boolean(window.cloudTrainingDisabled);
      $('iteration-runs').innerHTML=runs.length?runs.map((r,index)=>{
        const p=r.report;
        return `<details class="panel iteration-run" ${index===0||r.active?'open':''}><summary class="panel-heading"><h2>${r.mode==='model'?'监督模型与双阈值迭代':'双阈值校准'} <small class="mono">${esc(r.id)}</small></h2><span class="badge ${r.active?'low':r.status==='failed'?'high':'neutral'}">${r.active?'当前已应用':({queued:'排队中',running:'评估中',completed:'评估完成',failed:'未完成',interrupted:'已中断'}[r.status]||r.status)}</span></summary><div class="iteration-run-body"><p class="iteration-message">${esc(r.message||'')}</p>${p?`<p class="panel-footnote">复核样本使用：训练 ${p.feedback_usage.train} / 校准 ${p.feedback_usage.calibration} / 新增审计 ${p.feedback_usage.audit}；排除 ${p.excluded_samples.length} 件。<br>下阈值 ${fmt(p.thresholds.current[0])} → ${fmt(p.thresholds.candidate[0])}；上阈值 ${fmt(p.thresholds.current[1])} → ${fmt(p.thresholds.candidate[1])}</p>${metricsTable(p)}<p class="iteration-gate ${p.eligible_for_activation?'passed':''}">${p.eligible_for_activation?'发布检查通过，等待检验人员应用候选版本。':esc(p.gate_reasons.join('；'))}</p><details class="iteration-protocol"><summary>查看数据隔离与样本排除说明</summary><p>${esc(p.protocol)}</p><p>${esc(p.high_threshold_rule)}</p><p>${esc(p.scope)}</p>${p.excluded_samples.map(s=>`<p>${esc(s.session_id||'未知样本')}：${esc(s.reason)}</p>`).join('')}</details><div class="iteration-actions"><button class="button primary" data-activate="${esc(r.id)}" ${!p.eligible_for_activation||r.active?'disabled':''}>应用此候选版本</button>${r.active&&r.has_publication?`<button class="button secondary" data-rollback="${esc(r.id)}">回滚至发布前版本</button>`:''}</div>`:''}</div></details>`;
      }).join(''):'<div class="panel empty-state"><h3>尚无迭代任务</h3><p>先在“人工复核”中确认标签并批准用于迭代，再生成候选。</p></div>';
      if(running&&!$('page-iterations').hidden)poll=setTimeout(loadRuns,2500);
    }catch(e){if(request===runRequest)failure('iteration-error',e);}
  }
  function init(){
    $('review-refresh').addEventListener('click',loadQueue);
    for(const id of ['review-product','review-filter'])$(id).addEventListener('change',()=>{offset=0;loadQueue();});
    $('review-prev').addEventListener('click',()=>{offset=Math.max(0,offset-50);loadQueue();});
    $('review-next').addEventListener('click',()=>{offset+=50;loadQueue();});
    $('review-table').addEventListener('click',e=>{const b=e.target.closest('[data-review-day]');if(b)openDetail(b.dataset.reviewDay,b.dataset.reviewKey);});
    $('iteration-refresh').addEventListener('click',loadRuns);
    $('iteration-product').addEventListener('change',loadRuns);
    $('iteration-form').addEventListener('submit',async e=>{e.preventDefault();$('iteration-start').disabled=true;show('iteration-error',false);try{await api('/api/iterations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product:$('iteration-product').value,mode:$('iteration-mode').value})});await loadRuns();}catch(error){failure('iteration-error',error);$('iteration-start').disabled=false;}});
    $('iteration-runs').addEventListener('click',async e=>{const b=e.target.closest('[data-activate],[data-rollback]');if(!b)return;if(state.busy||state.batchBusy){toast('请等待当前检测批次完成后再切换模型版本。');return;}b.disabled=true;const run=b.dataset.activate||b.dataset.rollback;try{const result=await api(`/api/iterations/${encodeURIComponent(run)}/${b.dataset.activate?'activate':'rollback'}`,{method:'POST'});toast(result.note);await loadProducts();resetResult();await loadRuns();}catch(error){failure('iteration-error',error);b.disabled=false;}});
  }
  return {html,init,setProducts,loadQueue,loadRuns,openReference,showQueue,openQueue};
})();
