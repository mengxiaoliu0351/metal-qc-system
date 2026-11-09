'use strict';
window.ProductUI = (() => {
  let current = '', products = [], timer = null, uploading = false, section = 'config', request = 0;
  const roleNames = {bank:'正常建库',train:'有标签训练',calibration:'阈值校准',qualification:'独立资格验证',test:'封存测试',feedback:'后续核验样本'};

  function init() {
    const nav = $('maintenance-nav');
    const entry = document.createElement('button');
    entry.className = 'nav-item'; entry.dataset.page = 'products';
    entry.innerHTML = icon('plus', true) + '<span>研发设置</span>';
    entry.addEventListener('click', () => goPage('products'));
    nav.insertBefore(entry, nav.querySelector('button'));
    const page = document.createElement('section');
    page.id = 'page-products'; page.className = 'page'; page.hidden = true;
    page.innerHTML=heading('products','<button class="button secondary" id="catalog-refresh">刷新当前页</button>')+`
      <div class="settings-tabs" role="tablist" aria-label="研发设置分区">
        <button type="button" role="tab" id="settings-tab-config" aria-controls="settings-panel-config" aria-selected="true" data-settings-tab="config">检测配置</button>
        <button type="button" role="tab" id="settings-tab-samples" aria-controls="settings-panel-samples" aria-selected="false" tabindex="-1" data-settings-tab="samples">样本与建库</button>
        <button type="button" role="tab" id="catalog-family-link" aria-controls="settings-panel-research" aria-selected="false" tabindex="-1" data-settings-tab="research">研究资料</button>
      </div><div id="catalog-error" class="inline-error" role="alert" hidden></div>
      <section id="settings-panel-config" role="tabpanel" aria-labelledby="settings-tab-config"><section class="panel"><div class="panel-heading"><h2>已接入的检测配置</h2></div><p class="panel-footnote">日常检测仍选择螺母、螺丝、网格；这里列出系统内部配置。螺丝入口会自动匹配视图。</p><div id="runtime-config-list">正在读取配置…</div><button type="button" class="button secondary" id="settings-inspect">进入工件检验</button></section></section>
      <section id="settings-panel-samples" role="tabpanel" aria-labelledby="settings-tab-samples" hidden>
        <section class="panel catalog-work"><div class="panel-heading"><h2>选择要维护的产品</h2></div><label>已登记产品<select id="catalog-select"><option value="">正在读取…</option></select></label><p id="catalog-source" class="panel-footnote"></p>
        <details id="catalog-create-panel"><summary>登记新产品</summary><form id="catalog-create-form" class="catalog-form catalog-create-form">
          <label>产品名称<input name="name" required maxlength="80" placeholder="例如 M2 螺丝"></label><label>型号与规格<input name="sku" required maxlength="100" placeholder="来源资料中的型号或规格"></label>
          <label>拍摄视图<input name="view" required maxlength="80" placeholder="例如 头部俯视"></label><label>数据来源<input name="source" required maxlength="500" placeholder="来源地址或数据集名称"></label>
          <label>使用许可<input name="license" required maxlength="200" placeholder="来源注明的许可或授权"></label><button class="button secondary" type="submit">登记产品</button></form></details></section>
        <section class="panel catalog-work"><h2>1. 导入样本</h2><p id="catalog-next-step" class="management-scope"></p><details class="sample-count-details"><summary>查看各用途样本数量</summary><div id="catalog-counts"></div></details>
          <form id="catalog-upload-form" class="catalog-form" hidden><div class="catalog-row"><label>样本用途<select id="catalog-role">${Object.entries(roleNames).map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}</select></label><label>来源标签<select id="catalog-label"><option value="normal">正常</option><option value="defect">缺陷</option></select></label></div>
          <label>实物或来源组编号（来源提供时填写）<input id="catalog-group" maxlength="160" placeholder="同一实物的图片填写同一编号"></label><label>导入图片<input id="catalog-files" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" multiple required></label>
          <p class="panel-footnote">正常建库只使用正常图；各用途独立保存，封存测试不能回流训练。</p><button class="button secondary" type="submit">导入所选图片</button><p id="catalog-progress" role="status"></p></form></section>
        <section class="panel catalog-work"><h2>2. 建库与验证</h2><div id="catalog-build-controls" class="catalog-row" hidden><label>建库方式<select id="catalog-mode"><option value="normal_memory">正常样本记忆库</option><option value="global_binary">记忆库＋有标签整图判别</option></select></label><button class="button primary" id="catalog-build">开始建库与验证</button></div><p class="panel-footnote">已接入的产品请到模型迭代生成候选，不直接覆盖现有配置。</p><button type="button" class="text-button" id="settings-iterations">进入模型迭代 →</button></section>
        <section class="panel catalog-work"><h2>3. 查看结果与接入状态</h2><div id="catalog-builds"></div></section>
      </section><section id="settings-panel-research" role="tabpanel" aria-labelledby="catalog-family-link" hidden></section>`;
    $('main').appendChild(page);
    window.ManagementUI.attachSettings($('settings-panel-config'));
    $('settings-inspect').addEventListener('click',()=>goPage('detect'));
    $('settings-iterations').addEventListener('click',()=>goPage('iterations'));
    page.querySelector('.settings-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-settings-tab]');if(b)selectTab(b.dataset.settingsTab);});
    page.querySelector('.settings-tabs').addEventListener('keydown',e=>{
      const tabs=[...page.querySelectorAll('[data-settings-tab]')],index=tabs.indexOf(e.target);if(index<0)return;
      const next={ArrowLeft:(index+2)%3,ArrowRight:(index+1)%3,Home:0,End:2}[e.key];if(next===undefined)return;e.preventDefault();tabs[next].focus();selectTab(tabs[next].dataset.settingsTab);
    });
    $('catalog-refresh').addEventListener('click', load);
    $('catalog-select').addEventListener('change', () => {current = $('catalog-select').value; render();});
    $('catalog-create-form').addEventListener('submit', create);
    $('catalog-upload-form').addEventListener('submit', upload);
    $('catalog-build').addEventListener('click', build);
    $('catalog-builds').addEventListener('click', activate);
    window.FamilyUI?.init();
  }

  function failure(error) { $('catalog-error').textContent = error.message; show('catalog-error'); }

  function selectTab(key,refresh=true){
    if(!['config','samples','research'].includes(key))return;
    const changed=section!==key;section=key;clearTimeout(timer);
    for(const b of $('page-products').querySelectorAll('[data-settings-tab]')){const active=b.dataset.settingsTab===key;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;}
    for(const k of ['config','samples','research'])show('settings-panel-'+k,k===key);
    if(changed&&refresh)load();
  }
  function open(key='config'){selectTab(key,false);goPage('products');}
  async function load(){clearTimeout(timer);show('catalog-error',false);if(section==='config')return loadConfig();if(section==='research')return window.FamilyUI.load();return loadSamples();}
  async function loadConfig(){
    const ticket=++request;
    try{const [data,e,policy]=await Promise.all([api('/api/products'),api('/api/inspection/evidence'),window.ManagementUI.loadPolicy()]);if(ticket!==request)return;
      const profiles=new Map((e.profiles||[]).map(p=>[p.product,p]));
      $('runtime-config-list').innerHTML='<div class="table-scroll"><table><thead><tr><th>内部配置</th><th>判别方式</th><th>自动处理权限</th><th>参数</th></tr></thead><tbody>'+(data.products||[]).map(p=>{
        const q=profiles.get(p.key),hold=q&&policy?.holds?.[`${p.key}:${q.model_key}`];
        const status=!policy?'当前权限待核对':hold?'已暂停，全部复核':!q?'未提供资格数据':q.allow_normal&&q.allow_isolate?'按门槛自动处理':!q.allow_normal&&!q.allow_isolate?'全部转人工复核':'部分自动处理';
        return `<tr><td>${esc(p.name)}</td><td>${esc(p.method||'本地感知')}</td><td><strong>${status}</strong></td><td><details><summary>查看</summary><p>路由下阈值 ${fmt(p.threshold)}<br>路由上阈值 ${fmt(p.t_high)}</p>${q?`<p>${esc(hold?.reason||q.reason||q.scope||'按当前版本权限处理')}</p>`:''}</details></td></tr>`;
      }).join('')+'</tbody></table></div>';
    }catch(e){if(ticket===request)failure(e);}
  }
  async function loadSamples(){
    const ticket=++request;
    try{const data=await api('/api/product-catalog');if(ticket!==request)return;products=data.products||[];
      if(!products.some(p=>p.key===current))current=products[0]?.key||'';
      $('catalog-select').innerHTML=products.length?products.map(p=>`<option value="${esc(p.key)}">${esc(p.name)} · ${esc(p.view)}</option>`).join(''):'<option value="">请先登记产品</option>';
      $('catalog-select').value=current;render();
      if(products.some(p=>p.builds.some(b=>['queued','running'].includes(b.status)))&&!$('page-products').hidden&&section==='samples')timer=setTimeout(loadSamples,2500);
    }catch(e){if(ticket===request)failure(e);}
  }

  function render() {
    const product = products.find(p=>p.key===current);
    show('catalog-upload-form', Boolean(product)); show('catalog-build-controls', Boolean(product) && !product.active_build);
    $('catalog-next-step').textContent=!product?'先登记或选择产品，再导入有来源标签的样本。':product.active_build?'该产品已接入检测；可补充样本，后续改进请进入模型迭代。':'按用途导入样本后，进行建库与验证。';
    $('catalog-source').textContent = product ? `${product.sku} · ${product.source} · ${product.license}` : '登记产品后可导入样本。';
    $('catalog-counts').innerHTML = product ? `<table class="catalog-table"><thead><tr><th>用途</th><th>正常</th><th>缺陷</th></tr></thead><tbody>${Object.entries(roleNames).map(([k,v])=>`<tr><td>${v}</td><td>${product.counts[k].normal}</td><td>${product.counts[k].defect}</td></tr>`).join('')}</tbody></table>` : '';
    $('catalog-builds').innerHTML = (product?.builds || []).slice(0,3).map(b=>{
      const r=b.report, q=r?.qualification, a=r?.automation;
      return `<article class="catalog-build"><details ${['running','queued','failed'].includes(b.status)?'open':''}><summary>${esc({queued:'等待建库',running:'正在建库',completed:'建库完成',failed:'建库失败',interrupted:'建库中断'}[b.status]||b.status)}</summary>
        <p>${esc(b.message)}</p>${r?`<p>校准 ${r.counts.calibration} 张 · 资格验证 ${r.counts.qualification} 张 · 封存测试 ${r.sealed_test_n} 张</p>
        <p>下阈值 ${fmt(r.threshold)} · 上阈值 ${fmt(r.t_high)}</p>
        <p>${q?`资格验证一致率 ${(q.accuracy*100).toFixed(1)}% · 缺陷召回 ${q.recall===null?'待补充缺陷样本':(q.recall*100).toFixed(1)+'%'}`:'尚无独立资格验证结果'}</p>
        <p>建库时资格检查：自动正常 ${a.allow_normal?'通过':'暂不启用'} · 自动隔离 ${a.allow_isolate?'通过':'暂不启用'}</p><p class="panel-footnote">${esc(a.rule)}</p>`:''}
        ${product.active_build===b.id&&product.runtime_automation?`<p role="status"><strong>当前自动处置：${product.runtime_automation.held?'已暂停，检测结果转人工复核':`正常归档${product.runtime_automation.allow_normal?'可用':'暂不启用'} / 隔离${product.runtime_automation.allow_isolate?'可用':'暂不启用'}`}</strong> ${esc(product.runtime_automation.reason)}</p>`:''}
        </details>${b.status==='completed'&&!product.active_build?`<button class="button primary" data-catalog-activate="${esc(b.id)}">接入检测工作台</button>`:''}
        ${product.active_build===b.id?'<span class="badge low">已接入检测</span><div class="catalog-row"><button class="button secondary" data-catalog-batch="test">运行封存测试</button><button class="button secondary" data-catalog-batch="feedback">准备核验演示样本</button></div>':''}</article>`;
    }).join('') || '<p class="panel-footnote">尚无建库任务，先完成样本导入。</p>';
  }

  async function create(event) {
    event.preventDefault(); show('catalog-error', false);
    const button = event.currentTarget.querySelector('button'); button.disabled=true;
    try {const p=await api('/api/product-catalog',{method:'POST',body:JSON.stringify(Object.fromEntries(new FormData(event.currentTarget)))}); current=p.key; event.currentTarget.reset(); await loadSamples(); $('catalog-create-panel').open=false; toast('产品已登记，请导入样本。');}
    catch(error){failure(error);} finally{button.disabled=false;}
  }

  async function upload(event) {
    event.preventDefault(); if(uploading||!current)return;
    const files=Array.from($('catalog-files').files); if(!files.length)return;
    const key=current, role=$('catalog-role').value, label=$('catalog-label').value, group=$('catalog-group').value;
    uploading=true; const button=event.currentTarget.querySelector('button'); button.disabled=true; show('catalog-error',false);
    try {for(let i=0;i<files.length;i++) {
      const form=new FormData(); form.append('file',files[i]);form.append('role',role);form.append('label',label);form.append('group_id',group);
      $('catalog-progress').textContent=`正在导入 ${i+1}/${files.length} · ${files[i].name}`;
      await api(`/api/product-catalog/${key}/samples`,{method:'POST',body:form});
    } $('catalog-progress').textContent=`已处理 ${files.length} 张图片，重复图片按内容合并。`; $('catalog-files').value='';}
    catch(error){failure(error);} finally{uploading=false;button.disabled=false;await load();}
  }

  async function build() {
    if(!current||uploading)return;
    $('catalog-build').disabled=true; show('catalog-error',false);
    try {await api(`/api/product-catalog/${current}/builds`,{method:'POST',body:JSON.stringify({mode:$('catalog-mode').value})}); await load();}
    catch(error){failure(error);} finally{$('catalog-build').disabled=false;}
  }

  async function activate(event) {
    const batch=event.target.closest('[data-catalog-batch]');
    if(batch){
      batch.disabled=true;show('catalog-error',false);
      try{const data=await api(`/api/product-catalog/${current}/batches`,{method:'POST',body:JSON.stringify({role:batch.dataset.catalogBatch})});goPage('batch');await window.BatchUI.open(data.id);}
      catch(error){failure(error);}finally{batch.disabled=false;}
      return;
    }
    const button=event.target.closest('[data-catalog-activate]'); if(!button)return;
    if(state.busy||state.batchBusy){toast('请等待当前检测结束后接入新产品。');return;}
    button.disabled=true;show('catalog-error',false);
    try {await api(`/api/product-catalog/${current}/builds/${button.dataset.catalogActivate}/activate`,{method:'POST'}); await loadProducts();await load();toast('新产品已加入工件检验与批次检测。');}
    catch(error){failure(error);button.disabled=false;}
  }
  return {init,load,selectTab,open};
})();
