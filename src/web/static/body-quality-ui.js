'use strict';
window.BodyQualityUI=(()=>{
  let current='screw',sequence=0,ready=false,currentNames;
  const names={normal_memory:'正常样本记忆库',global_binary:'有标签整图判别',fast_fusion:'折外融合判别'};
  const reasons={'independent qualification gate failed':'独立资格检查未通过','sealed test recall below floor':'封存测试缺陷召回率未达要求','sealed test normal false-positive rate above ceiling':'封存测试正常件误报率超过预设上限','sealed BODY view coverage below floor':'封存测试适用视图通过率未达要求','HEAD rejection below floor':'头部异视图排除率未达要求','historical regression gate failed':'历史998张回归检查未达要求'};
  const pct=n=>n===null?'—':(100*n).toFixed(1)+'%';
  function table(methods,selected,labels){return `<div class="family-table-scroll"><table class="catalog-table"><thead><tr><th>方法</th><th>准确率</th><th>缺陷召回</th><th>正常误报率</th><th>误报 / 漏检</th></tr></thead><tbody>${Object.entries(methods).map(([arm,m])=>`<tr><td>${esc(labels[arm]||arm)}${arm===selected?'（校准选择）':''}</td><td>${pct(m.accuracy)}</td><td>${pct(m.recall)}</td><td>${pct(m.false_positive_rate)}</td><td>${m.confusion_matrix[0][1]} / ${m.confusion_matrix[1][0]}</td></tr>`).join('')}</tbody></table></div>`;}
  function init(){
    $('body-quality-panel').innerHTML=`<details><summary>M2螺杆质量验证 · 来源OK／NG标签</summary><p id="body-quality-status" role="status">正在读取质量实验…</p><div id="body-quality-metrics"></div>
      <form id="body-quality-form" class="catalog-form" hidden><label>上传单件M2螺杆图片<input id="body-quality-file" type="file" accept="image/png,image/jpeg,image/bmp,image/webp" required></label><button id="body-quality-submit" class="button primary" type="submit">运行螺杆质量研究试检</button></form>
      <div id="body-quality-examples" class="family-actions"></div><div id="body-quality-error" class="inline-error" role="alert" hidden></div><div id="body-quality-result" hidden></div>
      <p class="panel-footnote">只适用于来源M2螺杆视图。与上方无质量标签的三类外观匹配分开验证；研究结论不归档、不生成反馈训练标签，需人工核验。</p><p class="panel-footnote">样例来源：Rizki Putra Prastio、Rodik Wahyu Indrawan，<a href="https://data.mendeley.com/datasets/2y6mcsyrpy/2" target="_blank" rel="noopener">M2 screw defect detection dataset V2</a>，<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a>。样例转为PNG；试检图另作候选裁剪、补边及热力图叠加。</p></details>`;
    $('body-quality-form').addEventListener('submit',async event=>{event.preventDefault();if(!ready)return;const file=$('body-quality-file').files[0];if(!file)return;const form=new FormData();form.append('file',file);await run(`/api/product-families/${encodeURIComponent(current)}/body-quality/detect`,{method:'POST',body:form});});
    $('body-quality-examples').addEventListener('click',async event=>{const button=event.target.closest('[data-body-example]');if(button&&ready)await run(`/api/product-families/${encodeURIComponent(current)}/body-quality/examples/${encodeURIComponent(button.dataset.bodyExample)}/detect`,{method:'POST'});});
  }
  async function load(key){
    current=key;ready=false;const ticket=++sequence;show('body-quality-form',false);show('body-quality-result',false);show('body-quality-error',false);$('body-quality-metrics').innerHTML='';$('body-quality-examples').innerHTML='';
    try{
      const d=await api(`/api/product-families/${encodeURIComponent(key)}/body-quality`);if(ticket!==sequence)return;
      if(!d.ready){$('body-quality-status').textContent=d.message;return;}
      ready=d.trial_ready;show('body-quality-form',ready);const r=d.report;currentNames=r.method_names||names;
      $('body-quality-status').textContent=`${r.version} · 封存测试${r.test_count}张（正常${r.test_normal} / 缺陷${r.test_defect}） · ${ready?'通过本轮研究试检门槛':'未通过本轮门槛，保留结果，不启用试检'}。`;
      let history='';
      if(r.historical_regression){const h=r.historical_regression;history=`<h3>历史${h.n}张回归检查</h3>${table({prior:h.baseline,current:h.current},'',{prior:'V1冻结结果',current:'本轮候选'})}<p>以上历史图片已经查看，不是新的盲测；误报、漏检仍须满足本轮门槛。</p>`;}
      if(r.previous_report){const previous=r.previous_report;history+=`<details><summary>保留V1完整失败对照 · ${previous.test_count}张</summary>${table(previous.methods,previous.selected_mode,names)}<p>V1因正常件误报率超过10%上限未启用，不能被本轮新样本成绩覆盖。</p></details>`;}
      const note=r.method_note||'正常库只使用正常参考；另两种方法使用来源缺陷标签训练，融合采用折外预测，不能称整套零样本。';
      $('body-quality-metrics').innerHTML=`${r.test_partition_description?`<p>${esc(r.test_partition_description)}</p>`:''}${table(r.methods,r.selected_mode,currentNames)}<p>以上对全部封存BODY图片统计，未只挑可判定图片。螺杆视图通过率 ${pct(r.scope_coverage)}，头部异视图被错误接受 ${r.head_controls.incorrectly_accepted_as_body}/${r.head_controls.n}；所有输出仍需核验。</p><p>${esc(note)}${esc(r.trial_gate_reasons.map(reason=>reasons[reason]||reason).join('；'))}</p>${history}`;
      $('body-quality-examples').innerHTML=ready?d.examples.map((s,i)=>`<button class="button" type="button" data-body-example="${esc(s.id)}">${s.source_view==='head'?'异视图头部对照':s.source_label==='normal'?'来源正常样例':'来源缺陷样例'} ${i+1}</button>`).join(''):'';
    }catch(e){if(ticket===sequence)$('body-quality-status').textContent='螺杆质量结果暂不可用：'+e.message;}
  }
  async function run(url,options){
    const ticket=sequence;show('body-quality-error',false);show('body-quality-result',false);const buttons=[...$('body-quality-panel').querySelectorAll('button')];buttons.forEach(b=>b.disabled=true);
    try{const d=await api(url,options);if(ticket!==sequence)return;
      $('body-quality-result').innerHTML=`<h3>${d.scope_ok?(d.normal_defect_hint==='defect'?'缺陷线索，待核验':'正常线索，待核验'):'不符合M2螺杆视图，本次未作质量判定'}</h3>${d.scope_ok?`<p>${esc((currentNames||names)[d.method]||d.method)} · 分数 ${d.score.toFixed(4)} · 判定阈值 ${d.threshold.toFixed(4)}。模型分数不是实际合格概率。</p>`:''}<img class="family-heatmap" src="data:image/png;base64,${d.heatmap}" alt="补边后完整图的外观差异热力图"><p>${esc(d.heatmap_scope)}</p><p>${esc(d.scope)}</p>`;show('body-quality-result');
    }catch(e){if(ticket===sequence){$('body-quality-error').textContent=e.message;show('body-quality-error');}}finally{buttons.forEach(b=>b.disabled=false);}
  }
  return {init,load};
})();
