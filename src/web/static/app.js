'use strict';

// The browser only presents API results; thresholds and inference stay on the server.
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths = {
  scan:'M8 3H4a1 1 0 0 0-1 1v4m13-5h4a1 1 0 0 1 1 1v4M3 16v4a1 1 0 0 0 1 1h4m8 0h4a1 1 0 0 0 1-1v-4M7 12h10m-5-5v10',
  grid:'M3 3h7v7H3Zm11 0h7v7h-7ZM3 14h7v7H3Zm11 0h7v7h-7Z',
  layers:'m12 3 10 5-10 5L2 8Zm-10 9 10 5 10-5M2 17l10 5 10-5',
  clock:'M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  spark:'m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5ZM20 2v4m-2-2h4',
  upload:'M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5',
  arrow:'M4 12h16m-6-6 6 6-6 6', check:'m5 12 4 4L19 6',
  alert:'m10.3 4-8 14a2 2 0 0 0 1.7 3h16a2 2 0 0 0 1.7-3l-8-14a2 2 0 0 0-3.4 0ZM12 9v4m0 4h.01',
  file:'M14 2H5a1 1 0 0 0-1 1v18a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1V8Zm0 0v6h6M8 13h8m-8 4h5',
  refresh:'M20 7a9 9 0 1 0 1 8M20 3v5h-5', plus:'M12 5v14M5 12h14',
  zoom:'M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0m-2 5 6 6M7 10h6m-3-3v6',
  x:'m6 6 12 12M6 18 18 6', search:'M18 10.5a7.5 7.5 0 1 1-15 0 7.5 7.5 0 0 1 15 0m-2 5.5 5 5'
};
const icon = (name, small = false) => `<svg class="icon${small?' small':''}" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[name] || paths.scan}"/></svg>`;
const routes = {
  low_risk:{label:'低风险 · 本地快检',short:'低风险',tone:'low',symbol:'check',hint:'未发现明显异常线索',end:'本地'},
  vlm_review:{label:'灰区 · 深度复核',short:'灰区复核',tone:'gray',symbol:'search',hint:'VLM 分析与知识检索',end:'云端'},
  high_risk:{label:'高风险 · 本地提示',short:'高风险',tone:'high',symbol:'alert',hint:'疑似缺陷，按策略隔离与抽查',end:'本地'},
  input_review:{label:'输入待核验',short:'输入待核验',tone:'gray'},
  processing_error:{label:'处理异常',short:'处理异常',tone:'high'},
  general:{label:'通用视觉直检',short:'通用直检',tone:'general'},
  fast:{label:'本地快检（旧版）',short:'快检·旧版',tone:'neutral'},
  slow:{label:'深度分析（旧版）',short:'深度·旧版',tone:'neutral'}
};
const routeInfo = key => routes[key] || {label:'未记录路由',short:'未记录',tone:'neutral'};
const badge = key => {const r=routeInfo(key); return `<span class="badge ${r.tone}">${esc(r.short)}</span>`;};
const fmt = n => Number.isFinite(n) ? n.toFixed(4) : '—';
const imageSrc = b64 => `data:image/png;base64,${b64}`;
const pages = {detect:['工件检验','01 / INSPECTION','工件图像检验 · 自动处理 / 例外复核','scan'],batch:['批次检验','02 / BATCH','同一产品逐件检测，汇总本批次结果。','layers'],review:['人工复核','03 / REVIEW','核对最终标签与系统原始判定。','check'],history:['检验档案','04 / RECORDS','按样品、产品或风险路由回查原始检测记录。','file'],iterations:['模型迭代','05 / ITERATION','复核样本驱动的模型与阈值评估。','refresh'],dashboard:['质效分析','06 / STATISTICS','自动处理、人工负担、服务耗时与核验质量。','grid'],assistant:['辅助判读','07 / CONSULTATION','结合当前工件图像、检测记录和适用标准问答。','search']};
const heading = (key, action='') => `<div class="page-heading"><div class="heading-copy"><span class="eyebrow">${pages[key][1]}</span><h1>${pages[key][0]}</h1><p>${pages[key][2]}</p></div>${action}</div>`;
const button = (id,text,symbol='arrow',kind='secondary',disabled=false) => `<button id="${id}" class="button ${kind}"${disabled?' disabled':''}>${icon(symbol)}<span>${text}</span></button>`;
const empty = (symbol,title,text) => `<div class="empty-state">${icon(symbol)}<h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
const productSelect = id => `<div class="field"><label for="${id}">工件类型 / 检测配方</label><select id="${id}"><option value="">正在加载产品…</option></select></div>`;
const imageTile = (kind,title,text,sub) => `<figure class="image-tile"><figcaption><span><b class="channel-id">${kind==='original'?'A':'B'}</b>${title}</span><span id="${kind}-meta" class="mono">${kind==='original'?'NO IMAGE':'ANOMALY MAP'}</span></figcaption><button class="image-stage" id="${kind}-stage" aria-label="放大${title}" disabled><span class="viewport-reticle" aria-hidden="true"></span><span class="image-empty" id="${kind}-empty"><span class="empty-channel">${kind==='original'?'A':'B'}</span><strong id="${kind}-placeholder">${text}</strong><small>${sub}</small></span><img id="${kind}-image" alt="${title}" hidden><span class="zoom-hint">${icon('zoom')}</span></button></figure>`;
const routeRows = Object.entries(routes).filter(([k])=>['low_risk','vlm_review','high_risk'].includes(k)).map(([k,r])=>`<div class="route-row" data-route="${k}"><span class="route-symbol ${r.tone}">${icon(r.symbol,true)}</span><div><strong>${r.label}</strong><small>${r.hint}</small></div><span class="route-end">${r.end}</span></div>`).join('');

$('app').innerHTML = `${window.MetalQCLanding.html}<div id="workspace" hidden>
<header class="sidebar" aria-label="主导航">
  <a class="brand" href="#home" aria-label="返回系统首页"><span class="brand-mark">M<span>Q</span></span><span class="brand-name">金属表面检验<small>METALQC / INSPECTION SYSTEM</small></span></a>
  <nav aria-label="检验功能">${[['检验作业',['detect','batch','review']],['记录追溯',['history','dashboard']],['模型维护',['iterations']],['辅助工具',['assistant']]].map(([label,keys])=>`<div class="nav-group"><span class="nav-group-label">${label}</span>${keys.map(k=>`<button class="nav-item${k==='detect'?' active':''}" data-page="${k}" ${k==='detect'?'aria-current="page"':''}>${icon(pages[k][3],true)}<span>${pages[k][0]}</span></button>`).join('')}</div>`).join('')}</nav>
  <div class="sidebar-footer"><span class="status-dot" id="connection-dot"></span><span id="connection-text">正在连接服务</span></div>
</header>
<div class="main-shell">
  <div class="topbar"><div class="breadcrumb"><a href="#home" class="return-home">系统首页</a> <span>/</span><strong id="breadcrumb-title">工件检验</strong></div><div class="topbar-right"><span class="privacy-tag">AI自动处理 · 例外人工复核</span><time id="current-date"></time></div></div>
  <main id="main" tabindex="-1">
    <section id="page-detect" class="page" aria-label="工件检验">
      ${heading('detect','<button class="text-button" id="policy-summary">正在读取作业策略…</button>')}
      <div class="inspection-layout">
        <aside class="panel setup-panel" aria-label="检测配置与图像载入">
          <div class="panel-heading"><h2><span class="section-number">01</span>工件与输入</h2></div>
          <div class="input-controls">${productSelect('product-select')}</div>
          <details class="recipe-details"><summary>检测配置与阈值</summary><div class="threshold-info" id="threshold-info">正在读取产品配置…</div></details>
          <div class="setup-divider"><span>待检图像</span><span class="mono">INPUT</span></div>
          <input type="file" id="file-input" accept="image/png,image/jpeg,image/webp,image/bmp" hidden>
          <button class="drop-zone" id="drop-zone"><span class="upload-icon">${icon('upload')}</span><strong>载入工件图像</strong><small>点击选择或拖入<br>PNG / JPG / WEBP / BMP · ≤10 MB</small></button>
          <div id="file-selected" class="selected-file" hidden><img id="file-thumb" class="file-thumb" alt="所选图片缩略图"><div><strong id="file-name"></strong><small id="file-size"></small></div><button class="icon-button" id="clear-file" aria-label="清除所选图片">${icon('x')}</button></div>
          ${button('detect-button','执行检测','scan','primary',true)}
          <div id="detect-error" class="inline-error" role="alert" hidden></div>
          <div id="detect-progress" class="progress-message" role="status" hidden><span class="spinner"></span><div><strong>正在检测…</strong><small>本地分析 / 灰区视觉复核</small></div><span id="elapsed-time" class="mono">0 s</span></div>
          <div class="setup-divider sequence-title"><span>检验流程</span><span class="mono">SEQUENCE</span></div>
          <div class="workflow"><div class="step current" id="step-input"><span>01</span><strong>载入工件</strong><small>按产品加载检测配置</small></div><div class="step" id="step-analyze"><span>02</span><strong>分析与分流</strong><small>异常定位 / 三区间门控</small></div><div class="step" id="step-review"><span>03</span><strong>人工复核</strong><small>核对原图并归档报告</small></div></div>
          <p class="setup-footnote">离线图像输入<br>热力图辅助定位，不作尺寸测量</p>
        </aside>
        <section class="panel visual-panel" aria-label="工件图像观察窗">
          <div class="panel-heading viewport-heading"><h2><span class="section-number">02</span>表面观察窗</h2><div class="view-tools" role="group" aria-label="观察窗布局"><button data-view="compare" aria-pressed="true">双窗</button><button data-view="original" aria-pressed="false">原图</button><button data-view="heatmap" aria-pressed="false">热图</button></div></div>
          <div class="image-comparison" id="image-comparison" data-view="compare">${imageTile('original','工件原图','未载入工件','从上方载入图像，或选择演示试件')}${imageTile('heatmap','局部异常响应','尚无检测结果','执行检测后生成异常热力图')}</div>
          <div class="image-footer"><span>${icon('zoom',true)} 点击图像放大检查</span><span class="heat-legend">相对异常响应&nbsp; 低<i></i>高</span></div>
          <div class="specimen-library"><div class="library-heading"><strong>演示试件</strong><span>metal_nut · 真实数据集图像，点击载入</span></div><div class="sample-list">
            ${[['nut-good-012','正常螺母','good / 012'],['nut-scratch-020','浅划痕螺母','scratch / 020'],['nut-scratch-001','划痕螺母','scratch / 001']].map(([key,label,file])=>`<button class="sample-piece" data-sample="${key}" aria-pressed="false"><img src="/static/samples/${key}.png" alt="${label}" loading="lazy"><span><strong>${label}</strong><small class="mono">${file}</small></span><span class="sample-load" aria-hidden="true">↗</span></button>`).join('')}
          </div><p>用于操作演示；载入后需执行检测，风险分流以实际返回结果为准。</p></div>
        </section>
        <aside class="panel result-panel" aria-label="检测结果">
          <div class="panel-heading"><h2><span class="section-number">03</span>判读与处置</h2><span id="result-state" class="badge neutral">等待检测</span></div>
          <div class="score-summary"><div><span class="field-label">整图异常评分</span><div id="result-score" class="score-value mono">—<small> / SCORE</small></div></div><div class="time-summary">${icon('clock',true)}<span id="result-time">尚未检测</span></div></div>
          <div class="risk-meter" id="risk-meter"><div class="risk-track"><span></span><span></span><span></span><i id="score-marker" hidden></i></div><div class="risk-ticks"><span>低风险</span><span>灰区</span><span>高风险</span></div><div class="risk-thresholds mono" id="result-thresholds">t_low — <span>t_high —</span></div></div>
          <details class="route-explanation"><summary>查看风险分流规则</summary><div class="route-list" id="route-list">${routeRows}</div></details>
          <div id="analysis-empty" class="analysis-empty"><span class="empty-result-mark">—</span>等待分析结果<small>检测完成后自动给出处置；只有例外和抽查样本转人工。</small></div>
          <div id="analysis-content" hidden>

            <section id="disposition-card" class="disposition-card" aria-live="polite"><span id="processing-state" class="badge neutral"></span><p id="processing-reason"></p></section>
            <div class="analysis-heading"><span class="field-label">检测证据与AI描述</span></div><h3 id="result-verdict"></h3><p id="result-description" class="analysis-text"></p>
            <div class="suggestion"><span>建议处置</span><p id="result-suggestion"></p></div>
            <div class="result-actions">${button('review-button','处理复核任务','check','primary')}${button('next-file','载入下一件','upload','primary')}${button('report-button','导出报告','file','secondary')}${button('explain-button','询问 AI 助手','search')}</div>
            <details class="feedback-actions" id="feedback-details"><summary>反馈使用意见</summary><div class="feedback-heading"><strong>这次检测符合实际吗？</strong><span>先记录使用意见，真实标签仍需人工核验。</span></div><textarea id="feedback-comment" rows="2" maxlength="1000" placeholder="可选：记录误报、漏检或描述问题"></textarea><div class="feedback-buttons"><button id="feedback-up" aria-pressed="false">结果基本正确</button><button id="feedback-down" aria-pressed="false">结果有误，标记复核</button></div><p id="feedback-status" role="status">反馈不会直接修改模型或阈值。</p><button id="feedback-review" hidden>前往该样本人工复核 →</button></details>
            <details id="classification-section" class="result-disclosure" hidden><summary>缺陷类型线索</summary><div class="analysis-heading"><span class="field-label">缺陷归因</span><span id="classification-source" class="caption"></span></div><div id="classification-list"></div></details>
            <details id="reference-section" class="result-disclosure" hidden open><summary>知识检索依据</summary><div class="analysis-heading"><span class="field-label">知识卡 / 来源目录</span><span class="caption">来源与适用性需核对</span></div><div id="reference-list"></div></details>
          </div>
          <div class="human-note">${icon('alert',true)}<span>自动判断与人工标签分别记录。</span></div>
        </aside>
      </div>
    </section>
    <section id="page-batch" class="page" aria-label="批次检验" hidden>${heading('batch',button('export-batch','导出本批 CSV','file','secondary',true))}<section class="panel batch-input-panel"><div class="input-controls">${productSelect('batch-product-select')}${button('batch-button','执行批次检验','layers','primary',true)}</div><input type="file" id="batch-input" accept="image/png,image/jpeg,image/webp,image/bmp" multiple hidden><button class="drop-zone compact" id="batch-drop">${icon('upload')}<strong>载入同批次工件图像</strong><small>同一产品配置，逐件分析与记录</small></button><div id="batch-file-list" class="batch-file-list"></div><div id="batch-progress" class="progress-message" role="status" hidden></div></section><div class="section-heading"><h2>本批次结果<span id="batch-count" class="count-label">0</span></h2><span class="caption"><span id="batch-action-summary">自动处理与例外分流</span></span></div><div id="batch-results" class="batch-grid">${empty('layers','本批次尚无记录','载入工件后执行检验，逐件结果将在这里显示。')}</div></section>
    <section id="page-history" class="page" aria-label="检验档案" hidden>${heading('history',button('refresh-history','刷新档案','refresh'))}<section class="panel" id="history-list-panel"><div class="history-toolbar"><div class="search-field">${icon('search')}<input id="history-search" type="search" placeholder="按工件或产品检索" aria-label="搜索检测记录"></div><select id="history-route" aria-label="按路由筛选"><option value="">全部风险分流</option><option value="low_risk">低风险</option><option value="vlm_review">灰区复核</option><option value="high_risk">高风险</option></select><select id="history-disposition" aria-label="按处理状态筛选"><option value="">全部处理状态</option><option value="auto_normal">系统正常归档</option><option value="auto_isolated">疑似缺陷隔离</option><option value="needs_review">待人工处理</option><option value="human_confirmed">人工已确认</option><option value="legacy">历史未分流</option></select><span class="caption" id="history-count"></span></div><div class="table-scroll"><table><thead><tr><th>检验时间</th><th>工件 / 产品</th><th>异常评分</th><th>风险分流</th><th>处理状态 / AI意见</th><th></th></tr></thead><tbody id="history-table"></tbody></table></div></section><section id="history-detail" class="panel history-detail" aria-live="polite" hidden></section></section>
    <section id="page-dashboard" class="page" aria-label="检验统计" hidden>${heading('dashboard',button('refresh-dashboard','刷新统计','refresh'))}<div id="dashboard-error" class="inline-error" role="alert" hidden></div><div class="stats-grid">${[['today','今日检验','scan','次 · 最近 500 条记录内'],['total','检验记录','clock','条 · 最多统计最近 500 条'],['gray','灰区复核','search','次 · 路由至 VLM'],['products','工件配方','layers','类 · 独立记忆库与阈值']].map(([id,label,i,hint])=>`<article class="panel stat-card"><span>${label}</span>${icon(i)}<strong id="stat-${id}">—</strong><small>${hint}</small></article>`).join('')}</div><div class="dashboard-grid"><section class="panel"><div class="panel-heading"><h2>检验风险分流</h2><span class="caption">最近 500 条记录</span></div><div id="route-distribution" class="route-distribution"></div><p class="panel-footnote">旧版路由与通用直检单独计数，不计入三区间占比。</p></section><section class="panel"><div class="panel-heading"><h2>近期异常评分</h2><span class="caption">最近 20 条有分数的记录</span></div><div id="score-chart" class="score-chart"></div><p class="panel-footnote">不同产品需结合各自阈值解读，不能直接比较风险。</p></section></div><section class="panel"><div class="panel-heading"><h2>工件检测配置</h2><span class="caption">当前服务配置</span></div><div id="product-cards" class="product-cards"></div></section></section>
    <section id="page-assistant" class="page" aria-label="辅助判读" hidden>${heading('assistant',button('clear-chat','新建判读对话','plus'))}<section class="panel chat-panel"><div class="chat-panel-header"><span class="assistant-avatar">${icon('search')}</span><div><strong>工件判读助手</strong><span>检测上下文 · 附图问答</span></div><span class="badge neutral">云端服务</span></div><div id="chat-messages" class="chat-messages" role="log" aria-label="助手对话" aria-live="polite"></div><div class="chat-composer"><div id="chat-context" class="chat-context" hidden></div><form id="chat-form"><textarea id="chat-input" rows="1" aria-label="输入问题" placeholder="描述需复核的异常，或查询检测记录…（Shift + Enter 换行）"></textarea><button class="button primary" id="chat-send" type="submit" aria-label="发送问题">${icon('arrow')}</button></form><small>判读意见供参考；具体质量要求需核对适用标准。</small></div></section></section>
    ${window.ReviewUI.html}
    <footer class="workspace-footer"><span>METALQC <b>/</b> 金属表面缺陷检验</span><span>智能分流 <b>·</b> 例外复核 <b>·</b> 数据迭代</span></footer>
  </main>
</div></div>`;

// Keep the specimen input and examples together; the observation window only shows images.
const specimenLibrary=document.querySelector('.specimen-library');
const sampleDrawer=document.createElement('details');sampleDrawer.className='sample-drawer';
sampleDrawer.innerHTML='<summary>使用演示试件 <span>正常 / 浅划痕 / 划痕</span></summary>';
sampleDrawer.appendChild(specimenLibrary);document.querySelector('.setup-panel').appendChild(sampleDrawer);
const modelNote=document.createElement('details');modelNote.className='model-note';modelNote.innerHTML='<summary>检测方法与评分说明</summary><p id="perception-method" class="panel-footnote" hidden></p>';
document.querySelector('.result-panel').appendChild(modelNote);
const routeExplanation=document.querySelector('.route-explanation');document.querySelector('.result-panel').appendChild(routeExplanation);
const stationPanel=document.querySelector('.result-panel');
const stationResult=document.createElement('div');stationResult.id='station-result-pane';stationResult.setAttribute('role','tabpanel');stationResult.setAttribute('aria-labelledby','station-tab-result');
while(stationPanel.children.length>1)stationResult.appendChild(stationPanel.children[1]);
stationPanel.appendChild(stationResult);
stationPanel.querySelector('.panel-heading').insertAdjacentHTML('afterend','<div class="station-tabs" role="tablist" aria-label="工位侧栏"><button type="button" role="tab" aria-selected="true" id="station-tab-result" aria-controls="station-result-pane" data-station-tab="result">结果处理</button><button type="button" role="tab" aria-selected="false" id="station-tab-assistant" aria-controls="station-assistant-pane" data-station-tab="assistant">AI助手</button></div>');
const stationAssistant=document.createElement('div');stationAssistant.id='station-assistant-pane';stationAssistant.setAttribute('role','tabpanel');stationAssistant.setAttribute('aria-labelledby','station-tab-assistant');stationAssistant.hidden=true;stationPanel.appendChild(stationAssistant);
const sharedChat=document.querySelector('.chat-panel');
$('disposition-card').after(document.querySelector('.result-actions'));
const suggestionBlock=document.querySelector('.suggestion');
const suggestionDetails=document.createElement('details');suggestionDetails.className='suggestion';suggestionDetails.innerHTML='<summary>AI处置建议（参考）</summary>';suggestionDetails.appendChild($('result-suggestion'));suggestionBlock.replaceWith(suggestionDetails);
const resultExtras=document.createElement('details');resultExtras.className='result-extras';resultExtras.innerHTML='<summary>更多依据与使用反馈</summary>';
for(const element of [$('feedback-details'),$('classification-section'),modelNote,routeExplanation])resultExtras.appendChild(element);
$('analysis-content').appendChild(resultExtras);
function stationTab(tab){
  stationResult.hidden=tab!=='result';stationAssistant.hidden=tab!=='assistant';
  document.querySelectorAll('[data-station-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.stationTab===tab)));
  if(tab==='assistant')stationAssistant.appendChild(sharedChat);
}
document.querySelector('.station-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-station-tab]');if(b)stationTab(b.dataset.stationTab);});
const qualityPage=$('page-dashboard');
qualityPage.querySelector('.stats-grid').hidden=true;
qualityPage.querySelector('.page-heading').insertAdjacentHTML('afterend',`
  <div class="quality-overview">${[['automation-rate','自动处理覆盖率'],['human-rate','转人工比例'],['median-time','服务耗时中位数'],['pending-tasks','未完成复核任务']].map(([id,label])=>`<article class="panel"><span>${label}</span><strong id="${id}">—</strong></article>`).join('')}</div>
  <section class="panel quality-audit"><h2>质量与成本观察</h2><div id="quality-audit-values">尚无新策略记录</div><p class="panel-footnote">人工一致性仅统计已核验样本，不等于独立测试准确率；自动判断不会作为真实标签训练。</p></section>
  <details class="panel policy-settings"><summary>作业策略与自动处置依据</summary><form id="policy-form"><label>作业模式<select id="policy-mode"><option value="assisted">辅助作业：自动处理 + 例外复核</option><option value="validation">验证全检：逐件对照真实标签</option></select></label><label>稳定随机抽查比例<select id="policy-rate"><option value="0.05">5%</option><option value="0.1">10%</option><option value="0.2">20%</option><option value="0.5">50%</option><option value="1">100%</option></select></label><label>调整人员<input id="policy-operator" required maxlength="80" placeholder="姓名或工号"></label><button class="button secondary" type="submit">保存后续检验策略</button></form><p id="policy-context" class="panel-footnote"></p><div id="automation-evidence"></div></details>
`);
function percentage(value){return Number.isFinite(value)?(value*100).toFixed(1)+'%':'—';}
async function loadInspectionPolicy(){
  try{const p=await api('/api/inspection/policy');$('policy-summary').textContent=(p.mode==='validation'?'验证全检':'辅助作业')+' · 抽查 '+Math.round(p.sample_rate*100)+'%';$('policy-mode').value=p.mode;$('policy-rate').value=String(p.sample_rate);$('policy-context').textContent=p.gray_policy+'。相同图像、产品和模型版本的抽查选择保持稳定；策略变更仅作用于后续检验。';}
  catch(e){$('policy-summary').textContent='作业策略读取失败';}
}
function renderOperational(o){
  if(!o)return;
  $('automation-rate').textContent=percentage(o.automation_rate);$('human-rate').textContent=percentage(o.human_review_rate);$('median-time').textContent=Number.isFinite(o.median_seconds)?o.median_seconds.toFixed(2)+' s':'—';$('pending-tasks').textContent=o.pending_review;
  $('quality-audit-values').innerHTML=`<span>系统正常归档 <b>${o.automatic_normal}</b></span><span>疑似缺陷隔离 <b>${o.automatic_isolated}</b></span><span>抽查样本 <b>${o.audit_selected}</b></span><span>VLM调用 <b>${o.vlm_calls} / ${o.new_policy_records}（${percentage(o.vlm_call_rate)}）</b></span><span>已核验一致性 <b>${percentage(o.reviewed_agreement)} / ${o.labelled_comparisons}件</b></span><span>发现误报 / 漏检 <b>${o.false_positive} / ${o.false_negative}</b></span><span>服务耗时 P95 <b>${Number.isFinite(o.p95_seconds)?o.p95_seconds.toFixed(2)+' s':'—'}</b></span><span>人审耗时中位数 <b>${Number.isFinite(o.review_median_seconds)?o.review_median_seconds.toFixed(1)+' s':'—'}</b></span>`;
}
async function loadAutomationEvidence(){
  try{const data=await api('/api/inspection/evidence');$('automation-evidence').innerHTML='<div class="table-scroll"><table><thead><tr><th>产品</th><th>自动正常归档</th><th>自动隔离</th><th>冻结回归观察</th></tr></thead><tbody>'+data.profiles.map(p=>`<tr><td>${esc(p.product)}</td><td>${p.allow_normal?'已启用 / 分数 < '+fmt(p.auto_low):'未达门槛，转人工'}</td><td>${p.allow_isolate?'已启用 / 分数 > '+fmt(p.auto_high):'未达门槛，转人工'}</td><td>${p.regression?`${p.regression.n}件；自动正常区域漏检${p.regression.auto_normal_misses}件，隔离区域误报${p.regression.auto_isolation_false_positives}件`:esc(p.reason)}</td></tr>`).join('')+'</tbody></table></div><p class="panel-footnote">以上是有限公开数据上的回归门槛，不能视为新产线准确率保证；接入新产品或采集条件需重新验证。抽查确认误判会暂停相应产品当前版本自动处理。</p>';}
  catch(e){$('automation-evidence').textContent='评估依据暂不可用：'+e.message;}
}
$('policy-form').addEventListener('submit',async e=>{e.preventDefault();if(state.busy||state.batchBusy){toast('请等待当前检测完成后再调整策略。');return;}const b=e.currentTarget.querySelector('button');b.disabled=true;try{await api('/api/inspection/policy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:$('policy-mode').value,sample_rate:Number($('policy-rate').value),operator:$('policy-operator').value.trim()})});await loadInspectionPolicy();toast('策略已保存，后续检验生效。');}catch(error){toast(error.message);}finally{b.disabled=false;}});
$('classification-section').insertAdjacentHTML('beforeend','<p class="panel-footnote">类型概率仅辅助归因；风险路由由异常分数决定。</p>');

$('result-state').setAttribute('role','status');
const state = {products:[],file:null,fileUrl:null,last:null,busy:false,sampleLoading:false,batchBusy:false,batchFiles:[],batchResults:[],batchUrls:[],records:[],chat:[],chatBusy:false,context:null};
let toastTimer, historyRequest=0;
let sampleRequest=0;
const show = (id, visible=true) => {$(id).hidden=!visible;};
function toast(text){$('toast').textContent=text;show('toast');clearTimeout(toastTimer);toastTimer=setTimeout(()=>show('toast',false),4500);}
function connection(ok){$('connection-dot').className=`status-dot ${ok?'online':'offline'}`;$('connection-text').textContent=ok?'检测服务已连接':'服务连接失败';}
async function api(url,options){const response=await fetch(url,options);let data;try{data=await response.json();}catch(_){throw new Error(`服务未返回有效数据（HTTP ${response.status}）`);}if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:data.error||`请求失败（HTTP ${response.status}）`);if(data.ok===false)throw new Error(data.error||'服务暂不可用');return data;}
function showHome(updateHash=true){
  const returning=!$('workspace').hidden;
  $('landing-screen').hidden=false;$('workspace').hidden=true;
  document.body.dataset.screen='home';
  document.title='MetalQC · 双链路智能质检';
  document.querySelector('meta[name="theme-color"]').content='#080a1c';
  document.querySelector('.skip-link').href='#landing-main';
  if(updateHash&&location.hash!=='#home')history.pushState(null,'','#home');
  if(returning){const section=location.hash.slice(1);if(section==='capabilities'||section==='workflow')requestAnimationFrame(()=>$(section).scrollIntoView({block:'start'}));else window.scrollTo(0,0);}
}
function goPage(key,updateHash=true){
  if(!Object.hasOwn(pages,key)){showHome(updateHash);return;}
  const entering=$('workspace').hidden;
  $('landing-screen').hidden=true;$('workspace').hidden=false;
  document.body.dataset.screen='workspace';
  document.querySelector('meta[name="theme-color"]').content='#1b2933';
  document.querySelector('.skip-link').href='#main';
  document.querySelectorAll('.page').forEach(p=>{p.hidden=p.id!==`page-${key}`;});
  document.querySelectorAll('.nav-item').forEach(b=>{const active=b.dataset.page===key;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  $('breadcrumb-title').textContent=pages[key][0];document.title=`${pages[key][0]} · MetalQC`;
  if(updateHash&&location.hash!==`#${key}`)history.pushState(null,'',`#${key}`);
  window.scrollTo(0,0);
  if(entering)$('main').focus({preventScroll:true});
  if(key==='assistant')$('page-assistant').appendChild(sharedChat);if(key==='detect'&&!stationAssistant.hidden)stationAssistant.appendChild(sharedChat);
  if(key==='history')loadHistory();if(key==='dashboard')loadDashboard();
  if(key==='review'){window.ReviewUI.showQueue();window.ReviewUI.loadQueue();}if(key==='iterations')window.ReviewUI.loadRuns();
}
document.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>goPage(b.dataset.page)));
window.addEventListener('hashchange',()=>{const key=location.hash.slice(1);if(key==='main'&&!$('workspace').hidden)return;goPage(Object.hasOwn(pages,key)?key:'home',false);});
document.querySelector('.skip-link').addEventListener('click',e=>{e.preventDefault();const target=document.body.dataset.screen==='home'?$('landing-main'):$('main');target.focus({preventScroll:true});target.scrollIntoView({block:'start'});});
$('current-date').textContent=new Intl.DateTimeFormat('zh-CN',{year:'numeric',month:'2-digit',day:'2-digit',weekday:'short'}).format(new Date());

function currentProduct(){return state.products.find(p=>p.key===$('product-select').value);}
function thresholds(product){return {low:product?.threshold,high:product?.t_high};}
function updateThresholds(){const general=$('product-select').value==='vlm_general';const product=currentProduct();const {low,high}=thresholds(product);$('threshold-info').innerHTML=general?'通用直检直接请求云端视觉模型，不使用产品记忆库或异常分数。':`${esc(product?.method||'本地感知')}<br>产品独立阈值：<b>t_low ${fmt(low)}</b> <span> / </span> <b>t_high ${fmt(high)}</b> · 灰区请求 VLM`;$('result-thresholds').innerHTML=`t_low ${fmt(low)}<span>t_high ${fmt(high)}</span>`;show('risk-meter',!general);show('route-list',!general);show('perception-method',false);$('detect-button').disabled=!state.file||state.busy||state.sampleLoading||state.batchBusy||!$('product-select').value;}
async function loadProducts(){try{const data=await api('/api/products');state.products=data.products||[];const options=state.products.map(p=>`<option value="${esc(p.key)}">${esc(p.name)}</option>`).join('')+(data.fallback?`<option value="${esc(data.fallback.key)}">${esc(data.fallback.name)}</option>`:'');for(const id of ['product-select','batch-product-select'])$(id).innerHTML=options;connection(true);updateThresholds();renderProductCards();window.ReviewUI.setProducts(state.products);}catch(e){connection(false);$('threshold-info').textContent='无法加载产品，请确认本地服务已启动后刷新页面。';toast(e.message);}}
function renderProductCards(){$('product-cards').innerHTML=state.products.map(p=>`<article class="product-card"><h3>${esc(p.name)}</h3><span class="mono">${esc(p.key)}</span><div class="product-thresholds"><span>t_low <b>${fmt(p.threshold)}</b></span><span>t_high <b>${fmt(p.t_high)}</b></span></div></article>`).join('')||empty('layers','暂无产品配置','请检查产品记忆库配置。');}
function workflow(step){['input','analyze','review'].forEach((name,i)=>{$(`step-${name}`).className=`step${i===step?' current':i<step?' done':''}`;});}
function setImage(kind,src){const has=Boolean(src);show(`${kind}-image`,has);show(`${kind}-empty`,!has);$(`${kind}-stage`).disabled=!has;if(has)$(`${kind}-image`).src=src;else $(`${kind}-image`).removeAttribute('src');}
function resetResult(){stationTab('result');resetFeedbackUI();state.last=null;show('analysis-content',false);show('analysis-empty');show('detect-error',false);$('result-score').innerHTML='—<small> / SCORE</small>';$('result-state').className='badge neutral';$('result-state').textContent='等待检测';$('result-time').textContent='尚未检测';show('score-marker',false);document.querySelectorAll('.route-row').forEach(r=>r.classList.remove('selected'));setImage('heatmap',null);$('heatmap-placeholder').textContent='检测后生成热力图';updateThresholds();workflow(0);}
function validFile(file){return file && /\.(png|jpe?g|webp|bmp)$/i.test(file.name) && file.size>0 && file.size<=10*1024*1024;}
function chooseFile(file){if(state.busy||state.batchBusy)return;state.sampleLoading=false;if(!validFile(file)){toast('请选择不超过 10 MB 的 PNG、JPG、WEBP 或 BMP 图片。');return;}++sampleRequest;resetResult();sampleDrawer.open=false;document.querySelectorAll('[data-sample]').forEach(b=>b.setAttribute('aria-pressed','false'));if(state.fileUrl)URL.revokeObjectURL(state.fileUrl);state.file=file;state.fileUrl=URL.createObjectURL(file);$('file-name').textContent=file.name;$('file-size').textContent=`${(file.size/1024).toFixed(0)} KB · 已就绪`;$('file-thumb').src=state.fileUrl;show('file-selected');$('drop-zone').classList.add('has-file');$('drop-zone').querySelector('strong').textContent='更换工件图像';$('original-image').onload=()=>{$('original-meta').textContent=`${$('original-image').naturalWidth} × ${$('original-image').naturalHeight}`;};setImage('original',state.fileUrl);updateThresholds();}
function bindDrop(zoneId,inputId,onFiles){const zone=$(zoneId),input=$(inputId);zone.addEventListener('click',()=>{if(!state.busy&&!state.batchBusy)input.click();});zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('dragover');});zone.addEventListener('dragleave',()=>zone.classList.remove('dragover'));zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('dragover');if(!state.busy&&!state.batchBusy)onFiles(Array.from(e.dataTransfer.files));});input.addEventListener('change',()=>{onFiles(Array.from(input.files));input.value='';});}
bindDrop('drop-zone','file-input',files=>{if(files[0])chooseFile(files[0]);});
$('clear-file').addEventListener('click',()=>{if(state.busy||state.batchBusy)return;++sampleRequest;state.sampleLoading=false;state.file=null;if(state.fileUrl)URL.revokeObjectURL(state.fileUrl);state.fileUrl=null;document.querySelectorAll('[data-sample]').forEach(b=>b.setAttribute('aria-pressed','false'));show('file-selected',false);$('drop-zone').classList.remove('has-file');$('drop-zone').querySelector('strong').textContent='载入工件图像';setImage('original',null);$('original-meta').textContent='NO IMAGE';resetResult();});
$('product-select').addEventListener('change',()=>{++sampleRequest;state.sampleLoading=false;resetResult();});
function busyInputs(busy){['product-select','batch-product-select','clear-file','drop-zone','batch-drop','batch-button'].forEach(id=>$(id).disabled=busy);document.querySelectorAll('[data-sample]').forEach(b=>b.disabled=busy);$('detect-button').disabled=busy||state.sampleLoading||!state.file||!$('product-select').value;if(!busy)$('batch-button').disabled=!state.batchFiles.length||!$('batch-product-select').value;}
// View modes only change the optical workspace; they never alter inference data.
document.querySelector('.view-tools').addEventListener('click',e=>{const control=e.target.closest('[data-view]');if(!control)return;$('image-comparison').dataset.view=control.dataset.view;document.querySelectorAll('.view-tools [data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===control)));});
const sampleNames={'nut-good-012':'metal_nut_good_012.png','nut-scratch-020':'metal_nut_scratch_020.png','nut-scratch-001':'metal_nut_scratch_001.png'};
async function loadSample(key){
  if(!Object.hasOwn(sampleNames,key)||state.busy||state.batchBusy)return;
  if(!state.products.some(p=>p.key==='metal_nut')){toast('请等待产品配置加载完成后再载入演示试件。');return;}
  const request=++sampleRequest;state.sampleLoading=true;updateThresholds();
  try{
    const response=await fetch(`/static/samples/${key}.png`);
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const blob=await response.blob();
    if(request!==sampleRequest||state.busy||state.batchBusy)return;
    const file=new File([blob],sampleNames[key],{type:'image/png'});
    if(!validFile(file))throw new Error('试件图像为空或大小无效');
    $('product-select').value='metal_nut';
    state.sampleLoading=false;chooseFile(file);
    document.querySelector(`[data-sample="${key}"]`).setAttribute('aria-pressed','true');
    toast('已载入真实螺母演示图，请执行检测。');
  }catch(e){if(request===sampleRequest)toast(`试件载入失败：${e.message}`);}
  finally{if(request===sampleRequest){state.sampleLoading=false;updateThresholds();}}
}
document.querySelector('.sample-list').addEventListener('click',e=>{const b=e.target.closest('[data-sample]');if(b)loadSample(b.dataset.sample);});
async function detect(){if(!state.file||state.busy||state.sampleLoading||state.batchBusy)return;state.busy=true;resetResult();busyInputs(true);show('detect-progress');workflow(1);$('result-state').textContent='正在分析';const start=performance.now();$('elapsed-time').textContent='0 s';const timer=setInterval(()=>{$('elapsed-time').textContent=`${Math.floor((performance.now()-start)/1000)} s`;},500);try{const fd=new FormData();fd.append('file',state.file);fd.append('product',$('product-select').value);const data=await api('/api/detect',{method:'POST',body:fd});renderResult(data,(performance.now()-start)/1000);connection(true);}catch(e){$('detect-error').textContent=`检测未完成：${e.message}。请检查服务或网络后重试。`;show('detect-error');$('result-state').textContent='检测失败';$('result-state').className='badge high';workflow(0);}finally{clearInterval(timer);show('detect-progress',false);state.busy=false;busyInputs(false);}}
$('detect-button').addEventListener('click',detect);
function probabilityHtml(probs){return Object.entries(probs||{}).filter(([,v])=>Number.isFinite(v)).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([label,value])=>{const percent=Math.max(0,Math.min(100,value*100));return `<div class="probability-row"><span title="${esc(label)}">${esc(label)}</span><div class="probability-bar"><i style="width:${percent}%"></i></div><span class="mono">${percent.toFixed(1)}%</span></div>`;}).join('');}
function renderResult(data,totalTime){
  state.last=data;
  const disposition=data.disposition||{};const needsReview=Boolean(disposition.requires_review);
  $('processing-state').textContent=disposition.label||'未作业分流';$('processing-state').className='badge '+(needsReview?'gray':disposition.status==='auto_isolated'?'high':'low');
  $('processing-reason').textContent=disposition.reason||'';show('next-file',!needsReview);
  $('review-button').textContent=disposition.sampled?'完成抽查核验':'处理复核任务';
  const info=routeInfo(data.route);
  $('result-state').className=`badge ${info.tone}`;
  $('result-state').textContent=info.short;
  $('result-score').innerHTML=`${data.score==null?'—':fmt(data.score)}<small> / ${data.score==null?'无本地分数':'SCORE'}</small>`;
  $('result-time').textContent=`总耗时 ${totalTime.toFixed(1)} s`;
  const method=data.perception_method;
  $('perception-method').textContent=method?`${method}${Number.isFinite(data.baseline_score)?` · 原始局部分数 ${fmt(data.baseline_score)}`:''}。热力图用于局部定位，整图评分以当前模型阈值判读。`:'';
  show('perception-method',Boolean(method));
  show('analysis-content');show('analysis-empty',false);
  show('risk-meter',data.score!=null);show('route-list',Number.isFinite(data.score));
  document.querySelectorAll('.route-row').forEach(row=>row.classList.toggle('selected',row.dataset.route===data.route));
  const product=currentProduct();
  const low=data.threshold??product?.threshold,high=data.t_high??product?.t_high;
  $('result-thresholds').innerHTML=`t_low ${fmt(low)}<span>t_high ${fmt(high)}</span>`;
  if(Number.isFinite(data.score)&&low>0&&high>low){
    const s=data.score;
    const pos=s<low?(s/low)*33.333:s<=high?33.333+(s-low)/(high-low)*33.333:66.666+Math.min(1,(s-high)/high)*33.334;
    $('score-marker').style.left=`${Math.max(1,Math.min(99,pos))}%`;show('score-marker');
  }
  setImage('heatmap',data.heatmap?imageSrc(data.heatmap):null);
  if(!data.heatmap)$('heatmap-placeholder').textContent=data.route==='general'?'通用直检不生成本地热力图':'本次未生成热力图';
  const c=data.cognition||{};
  $('result-verdict').textContent=c.conclusion||'需人工复核';
  $('result-description').textContent=c.defect_desc||c.error||'未返回分析描述，请人工复核。';
  $('result-suggestion').textContent=c.suggestion||'请结合原图进行人工复核。';
  const supervised=data.supervised_class?.probs;
  const probs=supervised||data.classification||{};
  show('classification-section',Object.keys(probs).length>0);
  $('classification-source').textContent=supervised?'监督分类头':'零样本文本匹配';
  $('classification-list').innerHTML=probabilityHtml(probs);
  renderReferences(data.references||[],c.ref_ids||[]);
  if(c.citation_check?.ok===false){$('reference-list').insertAdjacentHTML('afterbegin','<p class="inline-error">引用校验失败：模型给出的引用 ID 不属于本次检索结果，已转人工核对。</p>');show('reference-section');}
  show('report-button',Boolean(data.session_id));
  show('review-button',Boolean(data.session_id)&&needsReview);
  show('explain-button',Boolean(data.heatmap));
  show('feedback-details',Boolean(data.session_id));show('feedback-up',Boolean(data.session_id));show('feedback-down',Boolean(data.session_id));
  workflow(2);
}
function renderReferences(refs,ids){
  const items=Array.isArray(refs)?refs:[],refIds=Array.isArray(ids)?ids:[];
  show('reference-section',items.length>0||refIds.length>0);
  const legacy=items.some(r=>String(r.id).startsWith('GB34528'));
  const note=legacy?'历史引用已停用，旧标准编号及页码不能作为检测依据。':'教学知识卡不是国标原文；标准目录仅核验编号与名称，具体条文与适用性待接入核验。';
  $('reference-list').innerHTML='<p class="panel-footnote">'+esc(note)+'</p>'+(items.length?items.map(r=>`<article class="reference-item"><strong>${esc(r.section||r.id)}</strong><br><small>${esc(r.standard)}${r.page!=null?' · P'+esc(r.page):''} · ${esc(r.id)}</small><p>${esc(r.text)}</p>${String(r.source_url||'').startsWith('https://openstd.samr.gov.cn/')?`<a href="${esc(r.source_url)}" target="_blank" rel="noopener">查看官方目录 ↗</a>`:''}</article>`).join(''):`<p class="caption">模型提供的引用 ID：${esc(refIds.join('、'))}。请核对来源。</p>`);
}
function exportReport(id){if(id)window.open(`/api/report/${encodeURIComponent(id)}`,'_blank','noopener');}
$('next-file').addEventListener('click',()=>$('file-input').click());
$('policy-summary').addEventListener('click',()=>goPage('dashboard'));
$('report-button').addEventListener('click',()=>exportReport(state.last?.session_id));
$('review-button').addEventListener('click',()=>{if(state.last?.session_id)window.ReviewUI.openReference({session_id:state.last.session_id});});
function resetFeedbackUI(){if(!$('feedback-status'))return;$('feedback-details').open=false;$('feedback-status').textContent='反馈不会直接修改模型或阈值。';$('feedback-comment').value='';$('feedback-up').setAttribute('aria-pressed','false');$('feedback-down').setAttribute('aria-pressed','false');show('feedback-review',false);}
async function feedback(label){
  if(!state.last?.session_id)return;
  const session=state.last.session_id,comment=$('feedback-comment').value.trim();
  $('feedback-up').disabled=$('feedback-down').disabled=true;
  try{
    const feedbackData=await api('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:session,label,comment})});
    if(state.last?.session_id!==session)return;
    if(feedbackData.disposition){state.last.disposition=feedbackData.disposition;$('processing-state').textContent=feedbackData.disposition.label;$('processing-state').className='badge '+(feedbackData.disposition.requires_review?'gray':'low');$('review-button').textContent='处理复核任务';$('processing-reason').textContent=feedbackData.disposition.reason;show('review-button',feedbackData.disposition.requires_review);show('next-file',!feedbackData.disposition.requires_review);}
    $('feedback-up').setAttribute('aria-pressed',String(label==='up'));
    $('feedback-down').setAttribute('aria-pressed',String(label==='down'));
    $('feedback-status').textContent=label==='up'?'已记录“结果基本正确”。尚未确认真实标签，模型和阈值均未改变。':'已标记为待复核，并保存困难样本。请核验真实标签；模型和阈值均未改变。';
    show('feedback-review');
    toast('反馈已保存；人工核验后才可批准用于候选迭代。');
  }catch(e){$('feedback-status').textContent=`反馈未保存：${e.message}`;toast(`反馈未保存：${e.message}`);}
  finally{$('feedback-up').disabled=$('feedback-down').disabled=false;}
}
$('feedback-up').addEventListener('click',()=>feedback('up'));$('feedback-down').addEventListener('click',()=>feedback('down'));
$('feedback-review').addEventListener('click',()=>{if(state.last?.session_id)window.ReviewUI.openReference({session_id:state.last.session_id});});
function zoom(kind){const img=$(`${kind}-image`);if(img.hidden)return;$('dialog-title').textContent=kind==='original'?'原始图像':'异常热力图';$('dialog-image').src=img.src;$('image-dialog').showModal();}
$('original-stage').addEventListener('click',()=>zoom('original'));$('heatmap-stage').addEventListener('click',()=>zoom('heatmap'));$('close-dialog').addEventListener('click',()=>$('image-dialog').close());$('image-dialog').addEventListener('click',e=>{if(e.target===$('image-dialog'))$('image-dialog').close();});

bindDrop('batch-drop','batch-input',files=>{const valid=files.filter(validFile);if(valid.length!==files.length)toast('已跳过格式不支持、为空或超过 10 MB 的图片。');if(!valid.length)return;state.batchFiles=valid;$('batch-file-list').innerHTML=`已选择 ${valid.length} 张图片<br>${valid.map(f=>`<span class="file-chip" title="${esc(f.name)}">${esc(f.name)}</span>`).join('')}`;$('batch-button').disabled=!$('batch-product-select').value;});
async function runBatch(){if(!state.batchFiles.length||state.busy||state.batchBusy)return;state.batchBusy=true;busyInputs(true);$('export-batch').disabled=true;state.batchUrls.forEach(URL.revokeObjectURL);state.batchUrls=[];state.batchResults=[];$('batch-results').innerHTML='';$('batch-count').textContent='0';show('batch-progress');const product=$('batch-product-select').value;try{for(let i=0;i<state.batchFiles.length;i++){const file=state.batchFiles[i];$('batch-progress').innerHTML=`<span class="spinner"></span><span>正在检测 ${i+1} / ${state.batchFiles.length}：${esc(file.name)}</span>`;const preview=URL.createObjectURL(file);state.batchUrls.push(preview);let data;try{const fd=new FormData();fd.append('file',file);fd.append('product',product);data=await api('/api/detect',{method:'POST',body:fd});}catch(e){data={ok:false,error:e.message};}state.batchResults.push({...data,name:file.name,preview});renderBatchCard(state.batchResults.length-1);$('batch-count').textContent=String(state.batchResults.length);}}finally{state.batchBusy=false;busyInputs(false);show('batch-progress',false);$('export-batch').disabled=!state.batchResults.length;const failures=state.batchResults.filter(r=>r.ok===false).length;$('batch-action-summary').textContent=`自动处理 ${state.batchResults.filter(r=>r.disposition?.auto_result).length} 件 · 转人工 ${state.batchResults.filter(r=>r.disposition?.requires_review).length} 件 · 请求失败 ${failures} 件`;toast(`已处理 ${state.batchResults.length} 张图片${failures?`，其中 ${failures} 张失败，可查看原因。`:'。'}`);}}
function renderBatchCard(index){const r=state.batchResults[index];const src=r.heatmap?imageSrc(r.heatmap):r.preview;const failed=r.ok===false;const article=document.createElement('article');article.className='panel batch-card';article.innerHTML=`<div class="batch-image"><img src="${src}" alt="${esc(r.name)}检测结果" loading="lazy"></div><div class="batch-card-body"><div class="batch-card-title"><strong title="${esc(r.name)}">${esc(r.name)}</strong>${failed?'<span class="badge high">检测失败</span>':badge(r.route)}</div><span class="score mono">${fmt(r.score)}</span><p><strong>${esc(failed?r.error:r.disposition?.label||'未作业分流')}</strong></p><p>${esc(r.disposition?.reason||'')}</p><p>${esc(r.cognition?.defect_desc||'')}</p>${!failed&&r.session_id&&r.disposition?.requires_review?`<button class="button primary" data-review-index="${index}">处理复核任务</button>`:''}${!failed&&r.session_id?`<button class="button secondary" data-report-index="${index}">${icon('file')}导出报告</button>`:''}</div>`;$('batch-results').appendChild(article);}
$('batch-button').addEventListener('click',runBatch);$('batch-results').addEventListener('click',e=>{const review=e.target.closest('[data-review-index]');if(review){window.ReviewUI.openReference({session_id:state.batchResults[Number(review.dataset.reviewIndex)]?.session_id});return;}const b=e.target.closest('[data-report-index]');if(b)exportReport(state.batchResults[Number(b.dataset.reportIndex)]?.session_id);});
function csvCell(value){let str=String(value??'');if(/^[=+@\-\t\r]/.test(str))str="'"+str;return '"'+str.replace(/"/g,'""')+'"';}
function exportCSV(){if(!state.batchResults.length)return;const rows=[['图片','异常分数','风险路由','处理状态','转人工原因','是否抽查','AI建议','描述','错误'],...state.batchResults.map(r=>[r.name,Number.isFinite(r.score)?r.score.toFixed(4):'',routeInfo(r.route).label,r.disposition?.label||'',r.disposition?.reason||'',r.disposition?.sampled?'是':'否',r.cognition?.conclusion||'',r.cognition?.defect_desc||'',r.error||''])];const blob=new Blob(['\ufeff'+rows.map(row=>row.map(csvCell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=`MetalQC_批量检测_${new Date().toISOString().slice(0,10)}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export-batch').addEventListener('click',exportCSV);

async function loadHistory(){show('history-list-panel');show('refresh-history');const request=++historyRequest;$('refresh-history').disabled=true;$('history-table').innerHTML='<tr><td colspan="6">正在读取检测记录…</td></tr>';$('history-detail').hidden=true;try{const data=await api('/api/history?limit=50');if(request!==historyRequest)return;state.records=data.records||[];renderHistory();}catch(e){if(request===historyRequest){$('history-table').innerHTML=`<tr><td colspan="6">${esc(e.message)}，请点击刷新重试。</td></tr>`;$('history-count').textContent='读取失败';}}finally{if(request===historyRequest)$('refresh-history').disabled=false;}}
function renderHistory(){const query=$('history-search').value.trim().toLowerCase(),route=$('history-route').value,processing=$('history-disposition').value;const matches=state.records.map((rec,index)=>({rec,index})).filter(({rec})=>(!query||`${rec.sample_name||''} ${rec.product_name||''}`.toLowerCase().includes(query))&&(!route||rec.gate?.route===route)&&(!processing||(processing==='needs_review'?(rec.current_disposition||{}).requires_review:(rec.current_disposition||{}).status===processing)));$('history-count').textContent=`${matches.length} 条 / 最近 ${state.records.length} 条`;$('history-table').innerHTML=matches.length?matches.map(({rec:r,index})=>`<tr><td class="mono">${esc((r.timestamp||'').slice(0,19).replace('T',' '))}</td><td><strong>${esc(r.sample_name||'未命名样品')}</strong><small>${esc(r.product_name||r.product_key||'未记录产品')}</small></td><td class="mono">${fmt(r.score)}</td><td>${badge(r.gate?.route)}</td><td><strong>${esc(r.current_disposition?.label||'历史未分流')}</strong><small>${esc(r.cognition?.conclusion||'未记录AI意见')}</small></td><td><button class="text-button" data-history-index="${index}">查看详情 →</button></td></tr>`).join(''):'<tr><td colspan="6">暂无匹配的检测记录。</td></tr>';}
const historyBack='<div class="review-detail-toolbar"><button class="button secondary" id="history-back" type="button">← 返回检验档案</button><span>历史详情 · 图像、检测结果与报告</span></div>';
$('history-detail').addEventListener('click',e=>{if(e.target.closest('#history-back')){++detailRequest;show('history-detail',false);show('history-list-panel');show('refresh-history');window.scrollTo(0,0);}});
let detailRequest=0;
async function historyDetail(index){const record=state.records[index];if(!record)return;const request=++detailRequest;show('history-list-panel',false);show('refresh-history',false);show('history-detail');$('history-detail').innerHTML=historyBack+'<span class="spinner"></span> 正在加载图像与分析…';try{const data=await api(`/api/history/detail?day=${encodeURIComponent(record._day||'')}&key=${encodeURIComponent(record._file||'')}`);if(request!==detailRequest)return;const r=data.record||{},c=r.cognition||{};$('history-detail').innerHTML=historyBack+`<div class="panel-heading"><h2>${esc(r.sample_name||'检测详情')}</h2>${badge(r.gate?.route)}</div><div class="detail-images">${[['input_img','原始图像'],['heatmap_img','异常热力图']].filter(([k])=>data[k]).map(([k,title])=>`<figure><figcaption>${title}</figcaption><img src="${imageSrc(data[k])}" alt="${title}"></figure>`).join('')}</div><div class="detail-meta"><span>产品<strong>${esc(r.product_name||'未记录')}</strong></span><span>异常分数<strong class="mono">${fmt(r.score)}</strong></span><span>下阈值<strong class="mono">${fmt(r.gate?.threshold)}</strong></span><span>上阈值<strong class="mono">${fmt(r.gate?.t_high)}</strong></span><span>AI 建议<strong>${esc(c.conclusion||'需人工复核')}</strong></span></div><p class="detail-analysis">${esc(c.defect_desc||'暂无描述')}\n${esc(c.suggestion||'')}</p><p class="panel-footnote">${esc(r.current_disposition?.label||'历史未分流')}：${esc(r.current_disposition?.reason||'系统判断与人工标签分别保存')}</p>`;$('history-detail').insertAdjacentHTML('beforeend',`<div class="review-link-row"><a class="button primary" href="/api/history/report?${new URLSearchParams({day:record._day,key:record._file})}" target="_blank" rel="noopener">导出 PDF 报告</a><button class="button secondary" id="history-open-review">人工最终复核</button></div>`);$('history-open-review').addEventListener('click',()=>window.ReviewUI.openReference({day:record._day,key:record._file}));$('history-detail').scrollIntoView({behavior:'smooth',block:'nearest'});}catch(e){if(request===detailRequest)$('history-detail').innerHTML=historyBack+`<div class="inline-error">详情加载失败：${esc(e.message)}</div>`;}}
$('refresh-history').addEventListener('click',loadHistory);$('history-search').addEventListener('input',renderHistory);$('history-route').addEventListener('change',renderHistory);$('history-disposition').addEventListener('change',renderHistory);$('history-table').addEventListener('click',e=>{const b=e.target.closest('[data-history-index]');if(b)historyDetail(Number(b.dataset.historyIndex));});

async function loadDashboard(){$('refresh-dashboard').disabled=true;show('dashboard-error',false);try{const [stats,historyData]=await Promise.all([api('/api/stats'),api('/api/history?limit=50')]);renderOperational(stats.operations);loadInspectionPolicy();loadAutomationEvidence();$('stat-today').textContent=stats.today??'—';$('stat-total').textContent=stats.total??'—';$('stat-gray').textContent=stats.vlm_review??'—';$('stat-products').textContent=stats.product_count??state.products.length;const entries=[['low_risk',stats.low_risk||0,'#6faa87'],['vlm_review',stats.vlm_review||0,'#c9ad69'],['high_risk',stats.high_risk||0,'#cb9580']];const total=entries.reduce((sum,e)=>sum+e[1],0);$('route-distribution').innerHTML=total?entries.map(([key,count,color])=>`<div class="distribution-row"><span>${routeInfo(key).short}</span><div class="distribution-track"><i style="width:${count/total*100}%;background:${color}"></i></div><strong>${count} / ${(count/total*100).toFixed(0)}%</strong></div>`).join('')+`<div class="other-count">其他路由 ${Math.max(0,(stats.total||0)-total)} 条</div>`:empty('grid','暂无三区间记录',`${stats.total||0} 条历史记录中尚无新版路由数据。`);renderScoreChart((historyData.records||[]).filter(r=>Number.isFinite(r.score)).slice(0,20).reverse());renderProductCards();}catch(e){$('dashboard-error').textContent=`数据读取失败：${e.message}`;show('dashboard-error');}finally{$('refresh-dashboard').disabled=false;}}
function renderScoreChart(records){if(!records.length){$('score-chart').innerHTML=empty('grid','暂无分数记录','完成一次检测后可查看近期分数。');return;}const width=520,height=200,left=38,top=12,bottom=28,plotH=height-top-bottom;const max=Math.max(.1,...records.map(r=>r.score))*1.15;let svg='';for(let i=0;i<=4;i++){const y=top+plotH*i/4;svg+=`<line x1="${left}" y1="${y}" x2="510" y2="${y}" stroke="#e9eeea"/><text x="29" y="${y+3}" text-anchor="end" font-size="9" fill="#95a99b">${(max*(1-i/4)).toFixed(2)}</text>`;}const step=(width-left-10)/records.length;records.forEach((r,i)=>{const h=Math.max(1,r.score/max*plotH),x=left+i*step+step*.22;const color={low_risk:'#7bab8d',vlm_review:'#cfb77a',high_risk:'#cd9b87'}[r.gate?.route]||'#a2b5b5';svg+=`<rect x="${x}" y="${top+plotH-h}" width="${Math.max(2,step*.56)}" height="${h}" rx="2" fill="${color}"><title>${esc(r.sample_name||'样品')} · ${esc(r.product_name||'')} · ${fmt(r.score)}</title></rect>`;});svg+=`<text x="${left}" y="195" font-size="9" fill="#95a99b">较早</text><text x="510" y="195" text-anchor="end" font-size="9" fill="#95a99b">最近</text>`;$('score-chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="最近 ${records.length} 条检测分数柱状图">${svg}</svg>`;}
$('refresh-dashboard').addEventListener('click',loadDashboard);

function chatWelcome(){$('chat-messages').innerHTML=`<div class="assistant-welcome"><h2>核对工件异常与判读依据</h2><p>从工件检验页附带热力图和结果，可进一步询问异常位置；也可查询产品配置与检验记录。</p><div class="prompt-suggestions">${[['查询今日检验记录','今天检测了多少张图片？'],['核对各产品检测阈值','目前支持哪些产品，各自阈值是多少？'],['查看近期检验情况','最近各产品的检测情况如何？']].map(([label,prompt])=>`<button data-prompt="${prompt}">${label}${icon('arrow',true)}</button>`).join('')}</div></div>`;}
function appendMessage(role,text,loading=false){$('chat-messages').querySelector('.assistant-welcome')?.remove();const el=document.createElement('div');el.className=`chat-message ${role}`;el.innerHTML=`<span class="message-label">${role==='user'?'你':'MetalQC Assistant'}</span><div class="message-content"></div>`;const content=el.querySelector('.message-content');if(loading)content.innerHTML='<span class="spinner"></span>';else content.textContent=text;$('chat-messages').appendChild(el);$('chat-messages').scrollTop=$('chat-messages').scrollHeight;return content;}
function updateChatContext(){show('chat-context',Boolean(state.context));$('chat-context').innerHTML=state.context?`<img src="${imageSrc(state.context.image_b64)}" alt="附带热力图"><span>已附带当前检测热力图与分析结果</span><button id="remove-context" aria-label="移除附图">×</button>`:'';}
async function sendChat(question){const q=(question??$('chat-input').value).trim();if(!q||state.chatBusy)return;state.chatBusy=true;$('chat-input').value='';$('chat-send').disabled=true;$('clear-chat').disabled=true;$('explain-button').disabled=true;appendMessage('user',q);const content=appendMessage('assistant','',true);const messages=[...state.chat,{role:'user',content:q}];const context=state.context;try{const data=await api('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages,...(context||{})})});const answer=data.answer||'未返回回答，请稍后重试。';content.textContent=answer;state.chat.push({role:'user',content:q},{role:'assistant',content:answer});}catch(e){content.textContent=`请求未完成：${e.message}。请重新发送问题。`;}finally{state.chatBusy=false;$('chat-send').disabled=false;$('clear-chat').disabled=false;$('explain-button').disabled=false;$('chat-messages').scrollTop=$('chat-messages').scrollHeight;}}
$('chat-form').addEventListener('submit',e=>{e.preventDefault();sendChat();});$('chat-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();sendChat();}});$('chat-messages').addEventListener('click',e=>{const b=e.target.closest('[data-prompt]');if(b)sendChat(b.dataset.prompt);});$('clear-chat').addEventListener('click',()=>{if(state.chatBusy)return;state.chat=[];state.context=null;updateChatContext();chatWelcome();});$('chat-context').addEventListener('click',e=>{if(e.target.closest('#remove-context')){state.context=null;updateChatContext();}});
$('explain-button').addEventListener('click',()=>{if(!state.last?.heatmap||state.chatBusy)return;const r=state.last,c=r.cognition||{};state.context={image_b64:r.heatmap,det_ctx:`产品：${r.product||''}；异常分数：${fmt(r.score)}；路由：${routeInfo(r.route).label}；建议：${c.conclusion||''}；描述：${c.defect_desc||''}`};updateChatContext();stationTab('assistant');$('chat-input').value='请结合当前工件图像，解释异常位置与判断依据。';$('chat-input').focus();});

window.ReviewUI.init();chatWelcome();loadInspectionPolicy();goPage(location.hash.slice(1)||'home',false);loadProducts();
