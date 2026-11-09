'use strict';
// Operations are read-only here. The policy form has one home in settings.
window.ManagementUI=(()=>{
  let dashboardRequest=0,detailRequest=0,policyDirty=false;
  const number=v=>Number.isFinite(v)?String(v):'—';
  const percent=v=>Number.isFinite(v)?(100*v).toFixed(1)+'%':'—';
  const seconds=(v,digits=2)=>Number.isFinite(v)?v.toFixed(digits)+' s':'—';
  const facts=items=>items.map(([label,value])=>`<div><dt>${label}</dt><dd>${value}</dd></div>`).join('');
  function init(){
    const page=$('page-dashboard');
    page.innerHTML=heading('dashboard','<button class="button secondary" id="refresh-dashboard">刷新数据</button>')+`
      <div id="dashboard-error" class="inline-error" role="alert" hidden></div>
      <p id="dashboard-scope" class="management-scope">读取作业统计…</p>
      <div class="quality-overview">
        ${[['automation-rate','自动判断占比'],['pending-tasks','待复核任务'],['vlm-rate','VLM调用占比'],['median-time','检测耗时中位数']].map(([id,label])=>`<article class="panel"><span>${label}</span><strong id="${id}">—</strong></article>`).join('')}
      </div>
      <div class="management-columns">
        <section class="panel"><div class="panel-heading"><h2>处理结果</h2><button class="text-button" id="dashboard-records">查看档案 →</button></div><dl id="quality-results" class="management-facts"></dl><button class="button primary" id="dashboard-pending">处理待复核任务</button></section>
        <section class="panel"><div class="panel-heading"><h2>人工核验发现</h2></div><dl id="quality-reviewed" class="management-facts"></dl><p class="panel-footnote">一致性只统计已核验样本，不是独立测试准确率；没有核验时显示“—”。</p></section>
      </div>
      <section class="panel management-audit"><div class="panel-heading"><h2>本期抽查</h2><button class="text-button" id="dashboard-policy">调整抽查规则 →</button></div><p id="audit-period-label"></p><dl id="audit-window-values" class="management-facts"></dl><button class="button secondary" id="dashboard-sampled">处理抽查任务</button></section>
      <details class="panel management-details" id="dashboard-details"><summary>查看运行明细</summary><div id="dashboard-detail-data">展开后读取灰区分流、调用次数与耗时。</div><p class="panel-footnote">按需调用率和任务数量不等于实测费用或人力节省。</p></details>`;
    $('refresh-dashboard').addEventListener('click',load);
    $('dashboard-records').addEventListener('click',()=>goPage('history'));
    $('dashboard-pending').addEventListener('click',()=>window.ReviewUI.openQueue('pending'));
    $('dashboard-sampled').addEventListener('click',()=>window.ReviewUI.openQueue('sampled'));
    $('dashboard-policy').addEventListener('click',()=>{window.ProductUI.open('config');$('policy-form').scrollIntoView({block:'center',behavior:'smooth'});});
    $('dashboard-details').addEventListener('toggle',()=>{if($('dashboard-details').open)loadDetails();});
  }
  function attachSettings(parent){
    parent.insertAdjacentHTML('beforeend',`<section class="panel policy-settings"><div class="panel-heading"><h2>自动归档抽查规则</h2></div><p class="panel-footnote">自动记录中的抽查与例外任务，共用人工复核队列。修改只作用于后续检验。</p><form id="policy-form"><label>目标抽查比例<select id="policy-rate">${[5,10,20,50,100].map(n=>`<option value="${n/100}">${n}%</option>`).join('')}</select></label><label>抽查周期<select id="policy-period"><option value="day">每日</option><option value="week">每周</option></select></label><label>调整人员<input id="policy-operator" required maxlength="80" placeholder="姓名或工号"></label><button class="button secondary" type="submit">保存规则</button></form><p id="policy-context" class="panel-footnote"></p><p id="policy-message" role="status"></p><div id="policy-error" class="inline-error" role="alert" hidden></div></section>`);
    $('policy-form').addEventListener('input',()=>{policyDirty=true;});
    $('policy-form').addEventListener('change',()=>{policyDirty=true;});
    $('policy-form').addEventListener('submit',savePolicy);
  }
  async function loadPolicy(){
    try{const p=await api('/api/inspection/policy');
      $('policy-summary').textContent=`辅助作业 · ${p.audit_period==='week'?'每周':'每日'}目标抽查 ${Math.round(p.sample_rate*100)}%`;
      if(!policyDirty){$('policy-rate').value=String(p.sample_rate);$('policy-period').value=p.audit_period||'day';}
      $('policy-context').textContent='同一周期内相同图像、产品和版本的选择保持稳定；小批次实际比例可能波动。';
      show('policy-error',false);return p;
    }catch(e){$('policy-summary').textContent='抽查规则暂不可用';$('policy-error').textContent=e.message;show('policy-error');}
  }
  async function savePolicy(event){
    event.preventDefault();if(state.busy||state.batchBusy){toast('请等待当前检测完成后再调整规则。');return;}
    const operator=$('policy-operator').value.trim();if(!operator){$('policy-operator').focus();return;}
    const b=event.currentTarget.querySelector('button');b.disabled=true;show('policy-error',false);$('policy-message').textContent='正在保存…';
    try{await api('/api/inspection/policy',{method:'POST',body:JSON.stringify({mode:'assisted',sample_rate:Number($('policy-rate').value),audit_period:$('policy-period').value,operator})});policyDirty=false;await loadPolicy();$('policy-message').textContent='规则已保存，后续检验生效。';}
    catch(e){$('policy-message').textContent='未保存，请重试。';$('policy-error').textContent=e.message;show('policy-error');}finally{b.disabled=false;}
  }
  async function load(){
    const ticket=++dashboardRequest;$('refresh-dashboard').disabled=true;show('dashboard-error',false);
    try{const [s,a]=await Promise.all([api('/api/stats'),api('/api/inspection/audit')]);if(ticket!==dashboardRequest)return;
      const o=s.operations||{};
      $('automation-rate').textContent=percent(o.automation_rate);$('pending-tasks').textContent=number(o.pending_review);$('vlm-rate').textContent=percent(o.vlm_call_rate);$('median-time').textContent=seconds(o.median_seconds);
      $('dashboard-scope').textContent=`最近500条记录中的 ${number(o.new_policy_records)} 条新策略作业记录；自动判断含抽查样本，待复核包含需要补核验的记录。`;
      $('quality-results').innerHTML=facts([['系统正常记录',number(o.automatic_normal)],['疑似缺陷隔离',number(o.automatic_isolated)],['转人工任务',number(o.sent_to_review)]]);
      $('quality-reviewed').innerHTML=facts([['已核验记录',number(o.reviewed_records)],['已核验一致性',`${percent(o.reviewed_agreement)}（${number(o.labelled_comparisons)}件可对照）`],['发现误报 / 漏检',`${number(o.false_positive)} / ${number(o.false_negative)}`]]);
      $('audit-period-label').textContent=`${a.period_key||'当前周期'} · ${a.audit_period==='week'?'每周':'每日'}抽查`;
      $('audit-window-values').innerHTML=facts([['自动记录',number(a.automatic)],['抽查入队',number(a.selected)],['待核验',number(a.selected_pending)],['已确认错误',number(a.confirmed_errors)]]);
      $('dashboard-pending').textContent=Number.isFinite(o.pending_review)?`处理待复核任务（${o.pending_review}）`:'打开复核队列';
      $('dashboard-sampled').textContent=Number.isFinite(a.selected_pending)?`处理抽查任务（${a.selected_pending}）`:'打开抽查队列';
      if($('dashboard-details').open)await loadDetails(s);else $('dashboard-detail-data').textContent='展开后读取灰区分流、调用次数与耗时。';
    }catch(e){if(ticket===dashboardRequest){$('dashboard-error').textContent='读取失败：'+e.message;show('dashboard-error');}}finally{if(ticket===dashboardRequest)$('refresh-dashboard').disabled=false;}
  }
  async function loadDetails(stats){
    const ticket=++detailRequest;
    try{const [s,g]=await Promise.all([stats?Promise.resolve(stats):api('/api/stats'),api('/api/inspection/gray-evidence')]);if(ticket!==detailRequest)return;const o=s.operations||{};
      $('dashboard-detail-data').innerHTML='<h3>调用与耗时</h3><dl class="management-facts">'+facts([['VLM调用',`${number(o.vlm_calls)} / ${number(o.new_policy_records)}`],['检测耗时P95',seconds(o.p95_seconds)],['核验耗时中位数',seconds(o.review_median_seconds,1)]])+'</dl><h3>灰区处理</h3><dl class="management-facts">'+facts([['灰区记录',number(g.total)],['自动判正常',number(g.auto_normal)],['自动隔离疑似缺陷',number(g.auto_defect)],['仍不确定转人工',number(g.uncertain)]])+'</dl>';
    }catch(e){if(ticket===detailRequest)$('dashboard-detail-data').textContent='明细暂不可用：'+e.message;}
  }
  return {init,attachSettings,load,loadPolicy};
})();
