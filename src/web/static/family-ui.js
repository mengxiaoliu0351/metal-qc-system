'use strict';
window.FamilyUI=(()=>{
  let ready=false, requestSequence=0, currentView='reports';
  const names={separate:'分别建模（已知来源）',pooled:'完全合并库',structured:'结构化统一库（自动匹配）'};
  let expertName='统一库＋自动匹配分组头（后验回放）';
  const conditions={original:'原图回归',dim:'模拟变暗',warm:'模拟偏暖'};
  const percent=n=>n===null?'—':(n*100).toFixed(1)+'%';
  function init(){
    const panel=document.createElement('section');panel.id='family-pilot';panel.className='panel';
    panel.innerHTML=`<div id="family-research-root"><div id="family-catalog"></div>
      <div class="research-views" role="group" aria-label="研究资料类型">${[['reports','评估资料'],['appearance','外观资料'],['trial','高级试检']].map(([k,label])=>`<button type="button" class="button secondary" data-research-view="${k}" aria-pressed="${k==='reports'}">${label}</button>`).join('')}</div>
      <p id="research-status" class="panel-footnote">历史研究与日常检验分别管理。</p>
      <section id="research-reports"><a class="button secondary" href="/api/product-families/screw/showcase/export" id="research-export">下载螺丝评估汇总 CSV</a>
        <p class="panel-footnote">历史离线评估不等于当前现场准确率；查看时请核对分区和版本。</p>
        <details id="family-benchmark"><summary>查看历史方案完整对照</summary><p id="family-status" role="status"></p><div id="family-metrics"></div><details><summary>各来源误报与漏检</summary><div id="family-details"></div></details><details><summary>正常参考库单独评分</summary><div id="family-memory"></div></details><div id="family-expert"></div><div id="family-fresh"></div><details id="family-prior" hidden><summary>保留的失败实验</summary><div id="family-prior-table"></div></details><p id="family-scope-note" class="panel-footnote">历史研究试检不自动归档；未知型号与工况需独立验证。</p></details></section>
      <section id="research-appearance" hidden><div id="family-appearance"></div></section>
      <section id="research-trial" hidden><p>以下为研发试验。正式单图与批次检验请使用工件检验入口。</p><div id="family-matching"></div><div id="body-quality-panel"></div>
        <details><summary>原有质量研究试检</summary><form id="family-form" class="catalog-form" hidden><label>仅用于已验证的近景/头部视图<input id="family-file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required></label><button id="family-detect" class="button secondary" type="submit">运行研究试检</button></form><div id="family-error" class="inline-error" role="alert" hidden></div><div id="family-result" hidden></div></details></section></div>`;
    $('settings-panel-research').appendChild(panel);$('family-form').addEventListener('submit',detect);
    window.AppearanceUI?.init();window.BodyQualityUI?.init();window.FamilyCatalog.init(()=>load(false));
    panel.querySelector('.research-views').addEventListener('click',e=>{const b=e.target.closest('[data-research-view]');if(b)selectView(b.dataset.researchView);});
  }
  function refreshView(){
    if(currentView==='appearance')window.FamilyCatalog.gallery();
    if(currentView==='trial'){const key=window.FamilyCatalog.selected();window.AppearanceUI?.load(key);window.BodyQualityUI?.load(key);}
  }
  function selectView(view){
    if(!['reports','appearance','trial'].includes(view))return;
    currentView=view;
    for(const key of ['reports','appearance','trial'])show('research-'+key,key===view);
    $('family-pilot').querySelectorAll('[data-research-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.researchView===view)));
    refreshView();
  }
  async function load(refresh=true){
    const sequence=++requestSequence;ready=false;show('family-result',false);show('family-error',false);show('family-form',false);
    try{
      if(refresh)await window.FamilyCatalog.load();
      const family=window.FamilyCatalog.selected();
      show('research-export',family==='screw');
      const d=await api('/api/product-family/benchmark?family='+encodeURIComponent(family));
      if(sequence!==requestSequence)return;ready=d.ready;
      show('family-form',ready);
      show('family-benchmark',ready);show('family-scope-note',ready);

      if(!ready){$('family-status').textContent=d.message;$('research-status').textContent=d.message;refreshView();return;}
      $('research-status').textContent='评估资料已读取；外观资料与高级试检按需打开，不参与日常作业统计。';
      refreshView();
      const r=d.report;
      expertName=r.new_retained_evaluation?'统一库＋自动匹配分组头（多光照版）':'统一库＋自动匹配分组头（后验回放）';
      $('family-expert').innerHTML='';$('family-fresh').innerHTML='';show('family-prior',false);
      $('family-status').textContent=`1 个统一参考库 · ${r.groups.length} 组来源/视图 · ${r.label_budget.training} 张参与判别训练 · ${r.label_budget.calibration} 张校准 · ${r.eligible_for_advisory_pilot?'共用判别头原图检查通过':'共用判别头未通过原图不回退检查'}。试检采用自动匹配分组头的工程回放，结果仅供核验。`;
      $('family-metrics').innerHTML=`<div class="family-table-scroll"><table class="catalog-table"><thead><tr><th>条件 / 方案</th><th>两来源平均准确率</th><th>两来源平均缺陷召回</th></tr></thead><tbody>${Object.keys(conditions).flatMap(c=>Object.keys(names).map(a=>{const m=r.metrics[c][a];return `<tr><td>${conditions[c]} · ${names[a]}</td><td>${percent(m.macro_accuracy)}</td><td>${percent(m.macro_recall)}</td></tr>`;})).join('')}</tbody></table></div>`;
      $('family-details').innerHTML=`<div class="family-table-scroll"><table class="catalog-table"><thead><tr><th>条件 / 方案 / 来源</th><th>图数</th><th>误报</th><th>漏检</th><th>准确率</th></tr></thead><tbody>${Object.keys(conditions).flatMap(c=>Object.keys(names).flatMap(a=>r.groups.map(g=>{const m=r.metrics[c][a].by_group[g.key];return `<tr><td>${conditions[c]} · ${names[a]} · ${esc(g.name)}</td><td>${m.n}</td><td>${m.false_positives}</td><td>${m.false_negatives}</td><td>${percent(m.accuracy)}</td></tr>`;}))).join('')}</tbody></table></div><p class="panel-footnote">以上是有标签整图判别对照，三方案使用相同训练图片。正常库独立评分的结果另存于实验报告，不能把监督收益归因于合并库。</p>`;
      $('family-memory').innerHTML=`<div class="family-table-scroll"><table class="catalog-table"><thead><tr><th>条件 / 方案 / 来源</th><th>误报</th><th>漏检</th><th>准确率</th><th>缺陷召回</th></tr></thead><tbody>${Object.keys(conditions).flatMap(c=>Object.keys(names).flatMap(a=>r.groups.map(g=>{const m=r.normal_only_memory[c][a][g.key];return `<tr><td>${conditions[c]} · ${names[a]} · ${esc(g.name)}</td><td>${m.false_positives}</td><td>${m.false_negatives}</td><td>${percent(m.accuracy)}</td><td>${percent(m.recall)}</td></tr>`;}))).join('')}</tbody></table></div><p class="panel-footnote">每组80张正常参考，分别选1024个代表特征。合并方案共享总计2048个特征，阈值由正常校准图确定。光照训练与校准范围以对应版本说明为准；各条件结果不能视为独立样本。</p>`;
      if(r.routed_expert_replay){$('family-expert').innerHTML=`<h3>工程补充：统一库＋自动匹配分组头</h3><p>共用一个视觉编码器和参考总库，按图片自动匹配两个轻量分组头。该方案在观察三臂结果后补充回放，没有重新训练或调整阈值。</p><div class="family-table-scroll"><table class="catalog-table"><thead><tr><th>条件 / 来源</th><th>准确率</th><th>缺陷召回</th><th>误报 / 漏检</th></tr></thead><tbody>${Object.keys(conditions).flatMap(c=>r.groups.map(g=>{const m=r.routed_expert_replay.metrics[c][g.key];return `<tr><td>${conditions[c]} · ${esc(g.name)}</td><td>${percent(m.accuracy)}</td><td>${percent(m.recall)}</td><td>${m.false_positives} / ${m.false_negatives}</td></tr>`;})).join('')}</tbody></table></div><p class="panel-footnote">这是后验工程回放，保留了分组头，不是完全共用一个判别头，也不是新增盲测。未知来源与视图仍需验证。</p>`;}
      if(r.new_retained_evaluation){
        const f=r.new_retained_evaluation;
        expertName='统一库＋自动匹配分组头（多光照版）';
        $('family-status').textContent=`版本 ${d.model.version} · 1 个统一参考库 · ${r.groups.length} 组来源/视图 · ${d.model.training_images} 张原始训练图 / ${d.model.training_views} 个训练视图 · 新增留出检查${f.fresh_gate_passed?'通过':'未全部通过'}；研究试检，不自动归档。`;
        $('family-fresh').innerHTML=`<h3>多光照训练后的新增留出验证</h3><p>固定新增300张图片：MVTec正常100张，M2正常100张、缺陷100张。先固定图片，再训练光照版本；三种条件是同一批图片的相关视图。MVTec这部分没有新增缺陷，不能据此评价其缺陷召回。</p><div class="family-table-scroll"><table class="catalog-table"><thead><tr><th>条件 / 版本 / 来源</th><th>准确率</th><th>缺陷召回</th><th>误报 / 漏检</th></tr></thead><tbody>${Object.keys(conditions).flatMap(c=>['v1','v2'].flatMap(v=>r.groups.map(g=>{const m=f.metrics[v][c][g.key];return `<tr><td>${conditions[c]} · ${v==='v1'?'原图训练':'多光照训练'} · ${esc(g.name)}</td><td>${percent(m.accuracy)}</td><td>${percent(m.recall)}</td><td>${m.false_positives} / ${m.false_negatives}</td></tr>`;}))).join('')}</tbody></table></div><p class="panel-footnote">改进针对第一轮已发现的模拟光照问题；旧576张作为开发回归，新增300张与本轮拟合、校准、参考图片按像素去重。来源未提供实物身份，不代表独立厂家或产线。${esc(f.gate_reasons.join('；'))}</p>`;
        if($('family-expert')){$('family-expert').querySelector('p').textContent='共用一个视觉编码器和参考总库，按图片自动匹配轻量分组头。本版本在训练前已明确采用多光照样本；下表为原有576张开发回归，新增留出结果另列。';$('family-expert').querySelector('.panel-footnote').textContent='本版预先定义自动分组头方案，训练和校准包含三种光照视图；共有640张原始训练图，不把其1920个相关视图视为独立样本。保留轻量分组头，仍不代表陌生型号和真实环境的可靠性。';}
        if(r.prior_experiment){show('family-prior');$('family-prior-table').innerHTML=`<p>第一轮仅使用原图训练。它暴露了合并判别头的漏检增加和光照敏感问题，推动第二轮改进。</p><table class="catalog-table"><thead><tr><th>条件 / 方案</th><th>来源平均准确率</th><th>来源平均缺陷召回</th></tr></thead><tbody>${Object.keys(conditions).flatMap(c=>Object.keys(names).map(a=>{const m=r.prior_experiment.metrics[c][a];return `<tr><td>${conditions[c]} · ${names[a]}</td><td>${percent(m.macro_accuracy)}</td><td>${percent(m.macro_recall)}</td></tr>`;})).join('')}</tbody></table>`;}
      }
    }catch(e){if(sequence===requestSequence){show('family-benchmark',false);show('family-scope-note',false);$('research-status').textContent='研究资料暂不可用：'+e.message;}}
  }
  async function detect(event){
    event.preventDefault();if(!ready)return;
    const file=$('family-file').files[0];if(!file)return;
    const sequence=requestSequence,family=window.FamilyCatalog.selected();
    const form=new FormData();form.append('file',file);form.append('family',family);$('family-detect').disabled=true;show('family-error',false);
    try{
      const d=await api('/api/product-family/detect',{method:'POST',body:form});
      if(sequence!==requestSequence)return;
      $('family-result').innerHTML=`<h3>${esc(d.conclusion)} · 研究试检待核验</h3><p>自动匹配参考：${esc(d.routing.reference_name)} · 适用性初筛${d.routing.in_scope?'通过':'未通过'}</p>
        <p>${expertName} · 模型分数 ${d.scores.routed_expert.toFixed(4)} / 判定阈值 ${d.thresholds.routed_expert.toFixed(4)}</p>
        <details><summary>查看其他试验方法对照（不影响当前建议）</summary><table class="catalog-table"><thead><tr><th>方案</th><th>模型分数 / 判定阈值</th><th>图像判别线索</th></tr></thead><tbody>${['pooled','structured'].map(a=>`<tr><td>${names[a]}</td><td>${d.scores[a].toFixed(4)} / ${d.thresholds[a].toFixed(4)}</td><td>${d.predictions[a]==='defect'?'缺陷':'正常'}</td></tr>`).join('')}</tbody></table></details>
        <img class="family-heatmap" src="data:image/png;base64,${d.heatmap}" alt="统一库按自动匹配参考生成的外观差异热力图">
        <p class="panel-footnote">${esc(d.scope)}。热力图用于观察外观差异，不能单独证明缺陷；模型分数不是实际错误概率。</p>`;
      show('family-result');
    }catch(e){if(sequence===requestSequence){$('family-error').textContent=e.message;show('family-error');}}finally{$('family-detect').disabled=false;}
  }
  return {init,load};
})();
