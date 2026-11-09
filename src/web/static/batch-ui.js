'use strict';
window.BatchUI = (() => {
  let current = '', timer = null, jobs = [], loading = false;
  const statusNames = {draft:'等待导入或启动',importing:'准备样本',queued:'排队中',running:'检测中',paused:'已暂停',completed:'已完成',completed_with_errors:'完成，含失败项目',failed:'准备失败',cancelled:'已终止'};

  function init() {
    const panel = document.createElement('section'); panel.className='panel catalog-work';
    panel.innerHTML=`<div class="panel-heading"><h2>后台批次任务</h2><button id="batch-jobs-refresh" class="button secondary">刷新任务</button></div>
      <label>选择批次<select id="batch-job-select"><option value="">暂无批次</option></select></label>
      <p id="batch-job-status" role="status">导入图片后创建任务，进度和结果保存在服务器。</p>
      <div id="batch-evaluation" hidden></div>
      <div class="catalog-row"><button id="batch-job-start" class="button primary" hidden>启动 / 继续</button>
        <button id="batch-job-pause" class="button secondary" hidden>暂停</button>
        <button id="batch-job-retry" class="button secondary" hidden>重试失败项目</button>
        <button id="batch-job-cancel" class="button secondary" hidden>终止批次，保留结果</button></div>
      <div id="batch-job-error" class="inline-error" role="alert" hidden></div>`;
    $('page-batch').querySelector('.batch-input-panel').after(panel);
    $('batch-jobs-refresh').addEventListener('click', load);
    $('batch-job-select').addEventListener('change',()=>open($('batch-job-select').value));
    for(const [id,action,body] of [['start','start',{}],['pause','pause',{}],['retry','start',{retry_failed:true}],['cancel','cancel',{}]]) {
      $('batch-job-'+id).addEventListener('click',()=>control(action,body));
    }
  }

  function failure(error){$('batch-job-error').textContent=error.message;show('batch-job-error');}

  async function load() {
    if(loading)return;loading=true;
    try {
      const data=await api('/api/batches');jobs=data.batches||[];
      if(!jobs.some(j=>j.id===current))current=jobs[0]?.id||'';
      $('batch-job-select').innerHTML=jobs.length?jobs.map(j=>`<option value="${esc(j.id)}">${esc(j.name)} · #${j.id.slice(0,8)} · ${esc(statusNames[j.status]||j.status)} · ${j.total} 张</option>`).join(''):'<option value="">暂无批次</option>';
      $('batch-job-select').value=current;
      if(current)await open(current);
    }catch(error){failure(error);}finally{loading=false;}
  }

  async function open(key) {
    if(!key)return;
    clearTimeout(timer);current=key;
    try {
      const job=await api(`/api/batches/${key}`);if(current!==key)return;
      const active=['queued','running','importing'].includes(job.status);
      state.batchBusy=active;busyInputs(active);
      const count=job.counts, scope={production:'检验作业',evaluation:'封存评估',simulation:'核验演示'}[job.purpose]||job.purpose;
      $('batch-job-status').textContent=`批次 #${job.id.slice(0,8)} · ${scope} · ${job.recipe.config?.name||job.product} · ${statusNames[job.status]||job.status} · 已完成 ${count.completed}/${job.total} · 失败 ${count.failed} · ${job.message}`;
      show('batch-evaluation',Boolean(job.evaluation));
      if(job.evaluation){const e=job.evaluation;$('batch-evaluation').innerHTML=`<table class="catalog-table"><tbody><tr><th>来源标签对照</th><td>正常 ${e.normal} 张 / 缺陷 ${e.defect} 张</td></tr><tr><th>本地有二分类结果</th><td>${e.local_classified} / ${e.scored}</td></tr><tr><th>本地误报 / 漏检</th><td>${e.local_false_positives} / ${e.local_false_negatives}</td></tr><tr><th>自动正常中的缺陷</th><td>${e.automatic_normal_misses}</td></tr><tr><th>自动隔离中的正常件</th><td>${e.automatic_isolation_false_positives}</td></tr></tbody></table><p class="panel-footnote">${esc(e.scope)}</p>`;}
      if(job.evaluation&&job.release_evaluation){const v=job.release_evaluation;$('batch-evaluation').insertAdjacentHTML('beforeend',`<p role="status"><strong>${v.automatic_release_vetoed?'该版本自动处置已否决 / 待核查':'本次未观察到自动判断错误'}</strong>${v.automatic_error_count!==undefined?` · ${v.automatic_error_count} 项`:''} ${esc(v.error||'封存结果只用于评价，不回流训练。')}</p>`);}
      show('batch-job-start',['draft','paused','failed'].includes(job.status));show('batch-job-pause',active);
      show('batch-job-retry',job.status==='completed_with_errors');show('batch-job-cancel',['draft','queued','running','paused','failed','importing'].includes(job.status));
      state.batchResults=job.items.filter(i=>i.result).map(i=>({...i.result,name:i.name,preview:`/api/batches/${key}/items/${i.id}/image`}));
      $('batch-results').innerHTML='';
      for(let i=Math.max(0,state.batchResults.length-100);i<state.batchResults.length;i++)renderBatchCard(i);
      $('batch-count').textContent=String(state.batchResults.length);
      $('batch-action-summary').textContent=`自动完成 ${count.automatic} 件 · 转人工 ${count.review} 件 · 深度分析尝试 ${count.vlm} 次 / 成功 ${state.batchResults.filter(r=>r.vlm_called&&r.cognition?.ok).length} 次${state.batchResults.length>100?' · 展示最近100件，可导出全部结果':''}`;
      $('export-batch').disabled=!state.batchResults.length;
      if(active)timer=setTimeout(()=>open(key),3000);
    }catch(error){failure(error);if(current===key)timer=setTimeout(()=>open(key),8000);}
  }

  async function control(action,body) {
    if(!current)return;show('batch-job-error',false);
    try{await api(`/api/batches/${current}/${action}`,{method:'POST',body:JSON.stringify(body)});await load();}
    catch(error){failure(error);}
  }

  async function start() {
    if(!state.batchFiles.length||state.busy||state.batchBusy)return;
    state.batchBusy=true;busyInputs(true);show('batch-progress');show('batch-job-error',false);
    try {
      const job=await api('/api/batches',{method:'POST',body:JSON.stringify({product:$('batch-product-select').value,name:'图片检验批次'})});current=job.id;
      for(let i=0;i<state.batchFiles.length;i++) {
        $('batch-progress').textContent=`正在保存图片 ${i+1}/${state.batchFiles.length}`;
        const form=new FormData();form.append('file',state.batchFiles[i]);
        await api(`/api/batches/${current}/items`,{method:'POST',body:form});
      }
      await api(`/api/batches/${current}/start`,{method:'POST',body:'{}'});
      state.batchFiles=[];$('batch-file-list').textContent='图片已保存为后台任务，刷新页面后可重新选择批次查看。';
      await load();
    }catch(error){failure(error);state.batchBusy=false;busyInputs(false);await load();}
    finally{show('batch-progress',false);}
  }
  return {init,load,open,start};
})();
