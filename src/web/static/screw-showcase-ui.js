'use strict';
window.ScrewShowcaseUI=(()=>{
  let sequence=0,current=null,tab='library';
  const pct=x=>Number.isFinite(x)?(100*x).toFixed(1)+'%':'—';
  function init(){
    const panel=document.createElement('section');panel.id='screw-showcase';panel.className='panel screw-showcase';
    $('page-dashboard').querySelector('.page-heading').insertAdjacentElement('afterend',panel);
    panel.innerHTML=`<header class="screw-showcase-header"><div><span class="eyebrow">SCREW FAMILY / DELIVERY</span><h2>螺丝族 · 从统一库到检验结果</h2><p>一次选择螺丝，自动匹配适用参考。检测能力与尚未通过的候选分别呈现。</p></div><a class="button primary" href="#family">进入螺丝检测 →</a></header>
      <nav class="screw-evidence-tabs" aria-label="螺丝族展示内容"><button type="button" data-screw-tab="library" aria-pressed="true">01 统一库与范围</button><button type="button" data-screw-tab="evaluation" aria-pressed="false">02 检测与评估</button><button type="button" data-screw-tab="workflow" aria-pressed="false">03 完整流程演示</button></nav>
      <div id="screw-showcase-status" role="status">读取螺丝族库与评估依据…</div><div id="screw-showcase-content"></div>`;
    panel.addEventListener('click',event=>{
      const button=event.target.closest('[data-screw-tab]');
      if(button){tab=button.dataset.screwTab;render();}
      const review=event.target.closest('[data-screw-review]');
      if(review)window.ReviewUI.openReference({session_id:review.dataset.screwReview});
    });
    $('refresh-dashboard').addEventListener('click',load);
  }
  function library(d){
    const i=d.inventory,e=d.evidence;
    return `<div class="screw-library-strip"><span><b>1</b> 族级参考目录</span><span><b>${i.view_groups.length}</b> 采集视图组</span><span><b>${i.appearance_reference_count}</b> 外观正向参考</span><span><b>${i.exclusion_reference_count}</b> 外观排除参考</span></div>
      <p>${esc(i.organization)}</p><p class="panel-footnote">共享编码器：${esc(i.shared_encoder)} · 分流版本：${esc(i.router_version)} · 外观版本：${esc(i.appearance_version)}</p>
      ${e?.construction?`<h3>正常库与判别配置怎样建立</h3><p>正常图建立局部特征库，用于差异热力图；有来源缺陷标签的拟合图训练轻量整图判别头；校准图确定阈值。外观资料不充当正常质量真值。</p><div class="table-scroll"><table><thead><tr><th>视图／版本</th><th>正常建库图</th><th>整图头拟合图</th><th>校准图</th><th>评价图</th><th>说明</th></tr></thead><tbody>${e.construction.map(c=>`<tr><td>${esc(c.view)}</td><td>${c.normal_reference_images}</td><td>${c.fitting_images}</td><td>${c.calibration_images}</td><td>${c.evaluation_images}</td><td>${esc(c.note)}</td></tr>`).join('')}</tbody></table></div><p class="panel-footnote">单位为图片；正常建库图可能同时用于拟合，不能把各列简单相加。候选配置未通过门槛时仍不接入检测。</p>`:''}
      <div class="screw-view-grid">${i.view_groups.map(v=>`<article><img src="${esc(v.reference_url)}" alt="${esc(v.name)}参考视图" loading="lazy"><div><h3>${esc(v.name)}</h3><strong class="${v.quality_model_bound&&!v.automatic_hold?'screw-status-ready':'screw-status-review'}">${esc(v.status)}</strong>${v.quality_model_bound?`<p>当前低／高阈值 ${v.threshold.toFixed(4)} / ${v.t_high.toFixed(4)}</p>`:'<p>识别到视图后进入复核，不输出质量分数。</p>'}</div></article>`).join('')}</div>
      <p>另有 ${esc(i.appearance_categories.join('、'))} 三类外观匹配参考。来源没有质量标签，匹配成功仍需核验，不能据此外观判断合格。</p>
      ${e?`<details class="screw-source-details"><summary>查看现有数据组成与用途</summary><div class="table-scroll"><table><thead><tr><th>来源／视图</th><th>已整理样本</th><th>质量依据</th></tr></thead><tbody>${e.datasets.map(s=>`<tr><td>${esc(s.name)}</td><td>${s.images.toLocaleString()} ${esc(s.unit)}${s.physical_specimens?` / ${s.physical_specimens}个实物`:''}${s.source_frames?` / ${s.source_frames}张原图`:''}</td><td>${s.quality_labels?'有来源质量标签':'仅外观类别，无质量标签'}</td></tr>`).join('')}</tbody></table></div><p>裁图、不同光照和不同实物是不同计数单位，不合并宣称独立样本总量。</p></details>`:'<p>评估证据包尚未安装，不显示示例成绩。</p>'}`;
  }
  function evaluation(d){
    const e=d.evidence;if(!e)return '<p>评估证据尚未就绪，暂不展示成绩。</p>';
    const r=e.routing,a=e.appearance;
    return `<div class="screw-evidence-note"><strong>先区分两种成绩</strong><p>匹配评估回答“参考是否适用”；质量评估回答“正常／缺陷判得怎样”。下表不计VLM或人工纠错，也不把拒判计为判对。</p></div>
      <div class="screw-routing-results"><div><h3>采集视图分流</h3><strong>${r.known_accepted} / ${r.known}</strong><p>已覆盖视图正确接受；未覆盖误接受 ${r.unknown_accepted} / ${r.unknown}</p><small>${d.evidence.routing_matches_current?'当前分流版本':'历史分流版本，当前已变化'} · 历史集成回归</small></div><div><h3>三类外观检索</h3><strong>${a.correct_accepted} / ${a.known}</strong><p>已覆盖外观正确接受；未覆盖误接受 ${a.unknown_accepted} / ${a.unknown}</p><small>${d.evidence.appearance_matches_current?'当前外观版本':'历史外观版本，当前已变化'} · 不代表缺陷检测</small></div></div>
      <h3>质量判别：逐视图、逐分区看结果</h3><div class="table-scroll"><table class="screw-evaluation-table"><thead><tr><th>配置／视图</th><th>评价范围</th><th>准确率</th><th>缺陷召回</th><th>正常误报率</th><th>误报／漏检</th><th>状态</th></tr></thead><tbody>${e.quality_cohorts.map(c=>`<tr><td><strong>${esc(c.name)}</strong></td><td>${esc(c.population)}<small>${esc(c.scope)}</small></td><td>${pct(c.metrics.accuracy)}</td><td>${pct(c.metrics.recall)}</td><td>${pct(c.metrics.false_positive_rate)}</td><td>${c.metrics.false_positives} / ${c.metrics.false_negatives}</td><td>${esc(c.state)}<small>${c.matches_current_model?'与当前模型及阈值匹配':'研究候选或历史结果'}</small></td></tr>`).join('')}</tbody></table></div>
      <details class="screw-source-details"><summary>混淆矩阵与可追溯依据</summary><div class="screw-matrices">${e.quality_cohorts.map(c=>{const m=c.metrics.confusion_matrix;return `<article><h4>${esc(c.name)}<small>${esc(c.population)}</small></h4><table aria-label="${esc(c.name)}混淆矩阵"><caption>行：真实标签 · 列：系统判定</caption><thead><tr><th></th><th>判正常</th><th>判缺陷</th></tr></thead><tbody><tr><th>真实正常</th><td>${m[0][0]}</td><td class="screw-error-count">${m[0][1]}</td></tr><tr><th>真实缺陷</th><td class="screw-error-count">${m[1][0]}</td><td>${m[1][1]}</td></tr></tbody></table></article>`;}).join('')}</div><p>证据版本：${esc(e.version)} · 汇总生成时间 ${esc(e.built_utc)}</p><p>成绩由原始混淆计数核对计算；源报告哈希保留在JSON中。</p><a href="/api/product-families/screw/showcase" target="_blank" rel="noopener">查看完整结构化依据</a></details>
      <ul class="screw-limitations">${e.limitations.map(x=>`<li>${esc(x)}</li>`).join('')}</ul><a class="button secondary" href="/api/product-families/screw/showcase/export">导出评估表（CSV）</a>`;
  }
  function workflow(d){
    const o=d.operations,a=d.integration;
    return `<ol class="screw-delivery-flow">${d.workflow.map(s=>`<li><span>${esc(s.step)}</span><div><h3>${esc(s.title)}</h3><p>${esc(s.detail)}</p></div></li>`).join('')}</ol>
      <div class="screw-demo-guide"><h3>建议按这个顺序演示</h3><p>① 工件检验选择“螺丝 · 自动匹配视图”，使用固定试件展示参考匹配、评分与热力图。</p><p>② 切换头部、螺杆或未覆盖外观，展示适用性与暂停规则如何转入复核。</p><p>③ 从检测结果打开复核入口和PDF报告；确认真实标签后，再进入模型迭代查看候选评估与应用／回滚。</p><p>④ 返回本页，对照质量结果和失败边界说明已实现的能力。</p><p class="panel-footnote">${esc(d.demo_note)}</p></div>
      <div class="screw-workflow-links"><a class="button primary" href="#family">开始单图检测</a><a class="button secondary" href="#batch">批次检验</a><a class="button secondary" href="#review">人工复核</a><a class="button secondary" href="#iterations">候选迭代</a><a class="button secondary" href="#history">档案与PDF</a></div>
      ${a?`<h3>已完成的公网流程验收</h3><p>以下为真实接口生成的演示记录，可直接打开核验入口和PDF。没有代填人工质量结论。${a.matches_current_workflow?'验收流程与当前核心代码一致。':'这是历史验收记录，当前核心流程代码已变化。'}</p><div class="table-scroll"><table><thead><tr><th>固定案例</th><th>系统动作</th><th>VLM</th><th>结果追溯</th></tr></thead><tbody>${a.checks.map(c=>`<tr><td>${esc(c.sample)}</td><td>${c.quality_model_bound?'匹配检测配置，执行原作业规则':'无质量模型，进入例外复核'}</td><td>${c.vlm_called?'已调用':'未调用'}</td><td><button class="button secondary" type="button" data-screw-review="${esc(c.session_id)}">打开复核记录</button> <a href="/api/report/${encodeURIComponent(c.session_id)}" target="_blank" rel="noopener">查看PDF</a></td></tr>`).join('')}</tbody></table></div><p>混合视图后台批次 ${a.mixed_view_count} / ${a.mixed_view_count} 项完成，批次编号 ${esc(a.batch_id)}。验收时间 ${esc(a.completed_utc)}。</p>`:''}
      <div class="screw-operations"><h3>螺丝族正式作业统计</h3><p>${o.new_policy_records} 条作业 · 自动处理 ${pct(o.automation_rate)} · 转人工 ${pct(o.human_review_rate)} · VLM调用 ${pct(o.vlm_call_rate)}</p><p>${esc(o.scope_note)}</p>${o.new_policy_records===0?'<p>当前暂无正式作业，演示记录已排除；不把演示成功包装成生产效率提升。</p>':''}</div>`;
  }
  function render(){
    if(!current)return;
    document.querySelectorAll('[data-screw-tab]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.screwTab===tab)));
    $('screw-showcase-content').innerHTML=({library,evaluation,workflow}[tab]||library)(current);
  }
  async function load(){
    const ticket=++sequence;
    try{const data=await api('/api/product-families/screw/showcase');if(ticket!==sequence)return;current=data;$('screw-showcase-status').textContent='当前配置与可追溯结果已载入；不会修改模型或作业权限。';render();}
    catch(error){if(ticket===sequence){current=null;$('screw-showcase-content').innerHTML='';$('screw-showcase-status').textContent='螺丝族依据暂不可用：'+error.message;}}
  }
  return {init,load};
})();
