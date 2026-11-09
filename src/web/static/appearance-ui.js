'use strict';
window.AppearanceUI=(()=>{
  let family='screw',ready=false,sequence=0;
  const pct=n=>n===null?'—':(100*n).toFixed(1)+'%';
  function init(){
    $('family-matching').innerHTML=`<section class="panel catalog-work"><div class="panel-heading"><h3>已审核外观 · 自动匹配</h3></div>
      <p id="appearance-status" role="status">正在读取外观匹配实验…</p>
      <form id="appearance-form" class="catalog-form" hidden><label>上传单件零件裁图<input id="appearance-file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required></label><button id="appearance-submit" class="button primary" type="submit">自动匹配外观参考</button></form>
      <p>无需手选亮面、黑色或机螺丝。请上传单件裁图；当前不支持多件合照自动定位。外观匹配不会给出合格／缺陷结论。</p>
      <div id="appearance-error" class="inline-error" role="alert" hidden></div><div id="appearance-result" hidden></div>
      <details id="appearance-benchmark" hidden><summary>查看外观匹配留出结果和范围</summary><div id="appearance-metrics"></div></details></section>`;
    $('appearance-form').addEventListener('submit',detect);
  }
  async function load(key){
    family=key;ready=false;const ticket=++sequence;show('appearance-form',false);show('appearance-result',false);show('appearance-error',false);show('appearance-benchmark',false);
    try{
      const d=await api(`/api/product-families/${encodeURIComponent(key)}/matching`);if(ticket!==sequence)return;
      ready=d.ready;show('appearance-form',ready);show('appearance-benchmark',ready);
      if(!ready){$('appearance-status').textContent=d.message;return;}
      const report=d.report,model=d.model,m=report.methods[report.selected_method].test;
      $('appearance-status').textContent=`${model.version} · 1 个外观参考文件 · ${report.reviewed_reference_count} 张人工审核可用图${report.exclusion_reference_count?' ＋ '+report.exclusion_reference_count+' 张来源类别排除参考':''} · ${model.categories.map(c=>c.name+' '+model.reference_counts[c.category_id]+'张').join(' / ')}。仅作外观检索实验。`;
      $('appearance-metrics').innerHTML=`<p>${report.primary_partition==='fresh_test'?'本轮新增封存 '+report.fresh_frame_count+' 张原图中的':'按原图分组留出'} ${m.n} 张单件图：已覆盖三类 ${m.known} 张，其余六类 ${m.other} 张。已覆盖类别强制选一类的准确率 ${pct(m.known_top1_accuracy)}；经过不确定门槛后，已覆盖图片接受率 ${pct(m.known_coverage)}；全部被接受图片匹配正确率 ${pct(m.accepted_precision_all)}；其余六类拒绝率 ${pct(m.other_rejection)}。未覆盖类型仍有 ${m.accepted_other} 张被误匹配，不能保证陌生类型都能被拒绝。</p>
        <div class="family-table-scroll"><table class="catalog-table"><thead><tr><th>外观</th><th>留出图</th><th>强制匹配正确</th><th>接受且正确</th><th>接受但错误</th><th>暂不匹配</th></tr></thead><tbody>${model.categories.map(c=>{const row=m.by_category[c.category_id];return `<tr><td>${esc(c.name)}</td><td>${row.n}</td><td>${row.correct_top1}</td><td>${row.accepted_correct}</td><td>${row.accepted_wrong}</td><td>${row.rejected}</td></tr>`;}).join('')}</tbody></table></div>
        <p class="panel-footnote">以上是离线单件裁图的外观匹配，不能称为缺陷检测准确率或完整系统验收。网页另有过小／空白图片检查。参考图来自用户审核；评价类别来自公开数据标注。模型方法与不确定门槛只用校准分区选择，测试分区没有参与调参；同一受控数据集，不代表未知厂家、真实产线或全部型号。</p>`;
      if(report.comparisons){
        const parts=['fresh_test','regression',...(report.comparisons.previous_fresh_regression?['previous_fresh_regression']:[])];
        $('appearance-metrics').innerHTML+=`<details><summary>查看原版与新版对照及旧数据回归</summary><table class="catalog-table"><thead><tr><th>分区 / 版本</th><th>已覆盖正确接受率</th><th>接受结果正确率</th><th>未覆盖误接受</th></tr></thead><tbody>${parts.flatMap(part=>['v1','v2'].map(v=>{const row=report.comparisons[part][v];return `<tr><td>${part==='fresh_test'?'新增封存评价':part==='regression'?'旧473张开发回归':'前轮301张开发回归'} / ${esc(report.comparison_labels?.[v]||v)}</td><td>${pct(row.known_correct_acceptance)}</td><td>${pct(row.accepted_precision_all)}</td><td>${row.accepted_other}</td></tr>`;})).join('')}</tbody></table><p>新增封存图来自同一公开数据集未使用的训练原图，不是外部产线盲测。旧数据用于发现问题，因此只称开发回归。类型排除参考来自类别标注，未作正常或缺陷核验。</p>${report.prior_candidate?'<p>第二版曾因旧数据新增1张已覆盖误匹配而拒绝应用；本版保持原有接受门槛，只增加类型排除比较。失败版本完整保留。</p>':''}</details>`;
      }
    }catch(e){if(ticket===sequence)$('appearance-status').textContent='外观匹配结果暂不可用：'+e.message;}
  }
  async function detect(event){
    event.preventDefault();if(!ready)return;const file=$('appearance-file').files[0];if(!file)return;
    const ticket=sequence,key=family,form=new FormData();form.append('file',file);$('appearance-submit').disabled=true;show('appearance-error',false);show('appearance-result',false);
    try{
      const d=await api(`/api/product-families/${encodeURIComponent(key)}/matching/detect`,{method:'POST',body:form});if(ticket!==sequence)return;
      $('appearance-result').innerHTML=`<h3>${d.matched?'匹配外观：'+esc(d.category_name):'未覆盖或匹配不确定'}</h3><p>参考相似度 ${d.similarity.toFixed(3)}；第一、第二类别差值 ${d.margin.toFixed(3)}。这些数值不是质量合格概率。</p><p>以下展示最接近的已审核参考；未通过门槛时仅用于比较。</p><div class="family-gallery">${d.references.map(r=>`<div class="family-sample"><a href="${esc(r.image_url)}" target="_blank" rel="noopener"><img src="${esc(r.image_url)}" alt="${esc(r.name)}已审核外观参考"></a><strong>${esc(r.name)}</strong><span>相似度 ${r.similarity.toFixed(3)}</span></div>`).join('')}</div>${d.exclusion_reference?`<p>${esc(d.reason)}。最接近的未覆盖类型：${esc(d.exclusion_reference.name)}，相似度 ${d.exclusion_reference.similarity.toFixed(3)}。</p><a href="${esc(d.exclusion_reference.image_url)}" target="_blank" rel="noopener">查看类型排除参考</a><p>${esc(d.exclusion_reference.provenance)}</p>`:''}<p>${esc(d.scope)}</p>`;show('appearance-result');
    }catch(e){if(ticket===sequence){$('appearance-error').textContent=e.message;show('appearance-error');}}finally{$('appearance-submit').disabled=false;}
  }
  return {init,load};
})();
