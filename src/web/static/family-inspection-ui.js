'use strict';
window.FamilyInspectionUI=(()=>{
  let key='screw',sequence=0,ready=false,busy=false,urls=[];
  const percent=n=>Number.isFinite(n)?(n*100).toFixed(1)+'%':'—';
  const seconds=n=>Number.isFinite(n)?n.toFixed(2)+' s':'—';
  function init(){
    $('family-inspection-panel').innerHTML=`<div class="panel-heading"><h2>产品族自动检验</h2><button id="family-inspection-refresh" class="button secondary" type="button">刷新作业统计</button></div>
      <p id="family-inspection-status" role="status">正在读取统一检验配置…</p><div id="family-inspection-views" class="family-inspection-views"></div>
      <form id="family-inspection-form" class="catalog-form" hidden><label>上传单件图片 · 一次可选最多10张<input id="family-inspection-files" type="file" accept="image/png,image/jpeg,image/bmp,image/webp" multiple required></label><button id="family-inspection-submit" class="button primary" type="submit">自动匹配并检测</button></form>
      <div id="family-inspection-examples" class="family-actions"></div><p class="panel-footnote">只需选择产品族，无需选择型号或视图。固定演示样例保留操作记录并转核验，不计入日常质效统计。请上传清晰单件图，多件自动定位尚未验证。</p>
      <p id="family-inspection-progress" role="status"></p><div id="family-inspection-error" class="inline-error" role="alert" hidden></div><div id="family-inspection-results"></div>
      <div id="family-inspection-metrics" class="family-inspection-summary"></div>
      <p class="panel-footnote">参考图仅说明采集视图，不能证明上传件合格。MVTec Software GmbH的<a href="https://www.mvtec.com/research-teaching/datasets/mvtec-ad" target="_blank" rel="noopener">MVTec AD</a>及<a href="https://www.mvtec.com/research-teaching/datasets/mvtec-screws" target="_blank" rel="noopener">MVTec Screws（CC BY-NC-SA 4.0）</a>；Rizki Putra Prastio、Rodik Wahyu Indrawan的<a href="https://data.mendeley.com/datasets/2y6mcsyrpy/2" target="_blank" rel="noopener">M2公开数据（CC BY 4.0）</a>。参考预览经过缩放，外观示例由来源标注框裁剪。</p>`;
    $('family-inspection-form').addEventListener('submit',async event=>{event.preventDefault();const files=[...$('family-inspection-files').files];if(files.length>10){error('一次最多10张，请分批提交');return;}await run(files.map(file=>({file,name:file.name})));});
    $('family-inspection-examples').addEventListener('click',async event=>{const button=event.target.closest('[data-family-example]');if(button)await run([{id:button.dataset.familyExample,name:button.textContent}]);});
    $('family-inspection-results').addEventListener('click',event=>{const button=event.target.closest('[data-family-review]');if(button)window.ReviewUI.openReference({session_id:button.dataset.familyReview});});
    $('family-inspection-refresh').addEventListener('click',metrics);
  }
  function error(message){$('family-inspection-error').textContent=message;show('family-inspection-error');}
  function clear(){urls.forEach(url=>URL.revokeObjectURL(url));urls=[];$('family-inspection-results').innerHTML='';$('family-inspection-progress').textContent='';}
  async function load(family){
    key=family;ready=false;const ticket=++sequence;clear();show('family-inspection-form',false);show('family-inspection-error',false);$('family-inspection-views').innerHTML='';$('family-inspection-examples').innerHTML='';$('family-inspection-metrics').innerHTML='';
    try{
      const d=await api(`/api/product-families/${encodeURIComponent(key)}/inspection`);if(ticket!==sequence)return;ready=d.ready;show('family-inspection-form',ready);
      if(!ready){$('family-inspection-status').textContent=d.message||'自动分流尚未通过验收，暂不执行质量检测';return;}
      $('family-inspection-status').textContent='选择一次产品族，系统自动匹配适用视图、调用已有检测流程；未覆盖与未通过门槛的情况进入复核。';
      $('family-inspection-views').innerHTML=d.views.map(v=>`<span><strong>${esc(v.name)}</strong> · ${esc(v.status)}</span>`).join('');
      $('family-inspection-examples').innerHTML=d.examples.map(s=>`<button class="button secondary" type="button" data-family-example="${esc(s.id)}">${esc(s.label)}</button>`).join('');
      await metrics();
    }catch(e){if(ticket===sequence){$('family-inspection-status').textContent='统一检验暂不可用';error(e.message);}}
  }
  async function metrics(){
    const ticket=sequence;try{const d=await api(`/api/product-families/${encodeURIComponent(key)}/inspection/metrics`);if(ticket!==sequence)return;const s=d.summary;
      $('family-inspection-metrics').innerHTML=[['本族作业',s.new_policy_records],['自动处理',percent(s.automation_rate)],['转人工',percent(s.human_review_rate)],['VLM调用',percent(s.vlm_call_rate)],['中位耗时',seconds(s.median_seconds)],['P95耗时',seconds(s.p95_seconds)]].map(([label,value])=>`<div><span>${label}</span><strong>${value}</strong></div>`).join('')+`<p class="panel-footnote">${esc(s.scope_note)}。暂无核验真值时不显示检测准确率。</p>`;
    }catch(e){if(ticket===sequence)$('family-inspection-metrics').textContent='作业统计暂不可用：'+e.message;}
  }
  function card(data,name,preview){
    const routing=data.family_routing||{},action=data.disposition||{};const picture=routing.reference_image_url;
    return `<article class="family-inspection-card"><h3>${esc(name)}</h3><p><strong>${esc(action.label||'未完成作业')}</strong> · ${esc(routing.view_name||'图像待核验')}</p>
      <div class="family-inspection-images">${preview?`<figure><figcaption>输入图片</figcaption><img src="${esc(preview)}" alt="上传工件"></figure>`:''}${picture?`<figure><figcaption>匹配视图参考 · 非合格证明</figcaption><img src="${esc(picture)}" alt="匹配视图的参考预览"></figure>`:''}${data.heatmap?`<figure><figcaption>局部差异热力图</figcaption><img src="data:image/png;base64,${data.heatmap}" alt="本地检测热力图"></figure>`:''}</div>
      <p>${esc(routing.reason||'')} ${esc(data.cognition?.defect_desc||'')}</p><p>${Number.isFinite(data.score)?'本地分数 '+data.score.toFixed(4)+' · ':''}VLM：${data.vlm_called?'已调用':'未调用'} · 总耗时 ${seconds(data.time_total)}</p>
      ${!data.quality_model_bound?'<p>未作正常／缺陷判断；本次记录可核验，但不能直接用于已有产品模型迭代。</p>':''}
      <p>${esc(action.reason||'')}</p><div class="family-actions">${data.session_id&&action.requires_review?`<button class="button primary" data-family-review="${esc(data.session_id)}" type="button">处理人工复核</button>`:''}${data.session_id?`<a class="button secondary" href="/api/report/${encodeURIComponent(data.session_id)}" target="_blank" rel="noopener">导出PDF报告</a>`:''}<a class="button secondary" href="#history">查看检验档案</a></div></article>`;
  }
  async function run(jobs){
    if(!ready||busy||!jobs.length)return;busy=true;clear();show('family-inspection-error',false);const family=key,ticket=sequence;const controls=[...$('family-inspection-panel').querySelectorAll('button,input')];controls.forEach(e=>e.disabled=true);
    try{for(let i=0;i<jobs.length;i++){
      if(ticket!==sequence)break;const job=jobs[i];$('family-inspection-progress').textContent=`正在处理 ${i+1} / ${jobs.length}：${job.name}`;
      let preview,url,options;if(job.file){if(job.file.size>10*1024*1024){error(job.name+' 超过10MB，已跳过');continue;}preview=URL.createObjectURL(job.file);urls.push(preview);const form=new FormData();form.append('file',job.file);url=`/api/product-families/${encodeURIComponent(family)}/inspection/detect`;options={method:'POST',body:form};}
      else{preview=`/api/product-families/${encodeURIComponent(family)}/inspection/examples/${encodeURIComponent(job.id)}`;url=preview+'/detect';options={method:'POST'};}
      try{const d=await api(url,options);if(ticket!==sequence)break;$('family-inspection-results').insertAdjacentHTML('beforeend',card(d,job.name,preview));}
      catch(e){if(ticket===sequence)error(job.name+'：'+e.message);}
    }if(ticket===sequence){$('family-inspection-progress').textContent='本轮处理结束；已有结果已保存到检验档案。';await metrics();}}
    finally{busy=false;controls.forEach(e=>e.disabled=false);}
  }
  return {init,load};
})();
