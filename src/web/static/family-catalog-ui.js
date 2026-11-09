'use strict';
window.FamilyCatalog=(()=>{
  let current='screw',changed,offset=0,sequence=0,rows=[];
  const labels={unreviewed:'外观资料待审核',accepted:'外观资料可用',rejected:'外观资料不适用'};
  function init(callback){
    changed=callback;
    $('family-catalog').innerHTML=`<div class="catalog-form"><label>研发资料所属产品族<select id="family-select"></select></label><p id="family-catalog-summary" role="status"></p></div>
      <details><summary>登记其他产品族</summary><form id="family-create" class="catalog-form"><label>产品族名称<input id="family-name" maxlength="80" required placeholder="例如：螺母族"></label><label>范围说明<input id="family-description" maxlength="1000" placeholder="结构、使用场景等"></label><button class="button" type="submit">登记产品族</button></form><p>登记后为待建设状态；需要独立的参考数据和配套评价结果，不会套用螺丝模型。</p></details>`;
    $('family-select').addEventListener('change',async()=>{current=$('family-select').value;offset=0;await load();changed();});
    $('family-create').addEventListener('submit',async event=>{
      event.preventDefault();const button=event.target.querySelector('button');button.disabled=true;
      try{const d=await api('/api/product-families',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('family-name').value,description:$('family-description').value})});current=d.family.key;await load();changed();}
      catch(e){$('family-catalog-summary').textContent=e.message;}finally{button.disabled=false;}
    });
    $('family-appearance').innerHTML=`<details id="family-gallery-panel"><summary>外观资料审核（不判断质量）</summary>
      <p id="family-data-summary"></p><p id="family-data-source"></p><p>先抽查三类、每类约5张：只看类别是否正确、零件是否裁剪完整、图片是否清晰。看不出缺陷不代表确认合格，无需寻找或标注缺陷。以下裁图依赖数据集标注框，不是自动定位结果。</p>
      <div class="catalog-form"><label>外观类别<select id="family-category"></select></label><label>资料审核状态<select id="family-quality"><option value="all">全部</option><option value="unreviewed">待审核</option><option value="accepted">外观资料可用</option><option value="rejected">外观资料不适用</option></select></label><label>审核人<input id="family-reviewer" maxlength="80" placeholder="填写本人姓名或代号"></label><label>备注（选填）<input id="family-review-note" maxlength="2000" placeholder="例如：裁剪截断、模糊、类别不符"></label></div>
      <div id="family-gallery" class="family-gallery"></div><p id="family-gallery-status" role="status"></p>
      <div class="family-actions"><button class="button" id="family-prev" type="button">上一页</button><button class="button" id="family-next" type="button">下一页</button><button class="button" id="family-select-train" type="button">勾选本页训练图</button><button class="button" id="family-clear-selection" type="button">取消勾选</button><button class="button" id="family-accept" type="button">类别正确、裁剪清晰</button><button class="button" id="family-reject" type="button">资料不适用</button><button class="button" id="family-reset" type="button">暂不确定</button></div>
      <p class="panel-footnote">审核只确认图片适合做外观识别资料，不赋予正常／缺陷标签、不加入正常参考库、不调整模型。原有训练、校准和测试分区保持固定。</p></details>`;
    ['family-category','family-quality'].forEach(id=>$(id).addEventListener('change',()=>{offset=0;gallery(false);}));
    $('family-prev').addEventListener('click',()=>{offset=Math.max(0,offset-24);gallery(false);});
    $('family-next').addEventListener('click',()=>{offset+=24;gallery(false);});
    $('family-select-train').addEventListener('click',()=>{$('family-gallery').querySelectorAll('.family-sample-check').forEach(e=>e.checked=e.dataset.split==='train');});
    $('family-clear-selection').addEventListener('click',()=>{$('family-gallery').querySelectorAll('.family-sample-check').forEach(e=>e.checked=false);});
    [['accepted','family-accept'],['rejected','family-reject'],['unreviewed','family-reset']].forEach(([status,id])=>$(id).addEventListener('click',()=>review(status)));
  }
  async function load(){
    const d=await api('/api/product-families');if(!d.families.some(f=>f.key===current))current=d.families[0]?.key||'screw';
    $('family-select').innerHTML=d.families.map(f=>`<option value="${esc(f.key)}">${esc(f.name)} · ${f.model_ready?'研究试检可用':'待建设'}</option>`).join('');$('family-select').value=current;
    const f=d.families.find(f=>f.key===current);
    $('family-catalog-summary').textContent=f?`${f.appearance_summary.types} 类候选外观 / ${f.appearance_summary.samples} 张候选图 · ${f.appearance_accepted||0} 张外观资料可用 / ${f.appearance_rejected||0} 张不适用 · 质量标签未提供${f.legacy_quality_marks?'；历史质量标记仅保留审计，不作参考批准依据':''}`:'尚无产品族';


  }
  async function gallery(reset=true){
    const ticket=++sequence,key=current;
    if(reset){offset=0;$('family-category').value='';$('family-quality').value='all';}
    $('family-gallery').innerHTML='';$('family-gallery-status').textContent='读取候选样本…';
    const category=reset?'':$('family-category').value;
    try{
      const d=await api(`/api/product-families/${encodeURIComponent(key)}/appearance?category=${encodeURIComponent(category)}&status=${encodeURIComponent($('family-quality').value)}&offset=${offset}&limit=24`);
      if(ticket!==sequence||key!==current)return;rows=d.samples;
      if(reset)$('family-category').innerHTML='<option value="">全部外观</option>'+d.types.map(t=>`<option value="${esc(t.category_id)}">${esc(t.name)}</option>`).join('');
      $('family-data-summary').textContent=`候选图 ${d.summary.samples} 张，覆盖 ${d.summary.types} 种外观；这些外观尚未纳入当前质量检测模型。`;
      $('family-data-source').textContent=d.dataset?`来源：${d.dataset.dataset}，${d.dataset.publisher}；许可：${d.dataset.license}。展示名称是外观别名，实际类别以原始 type 编号为准。`:'';
      if(d.dataset?.official?.startsWith('https://')){const link=document.createElement('a');link.href=d.dataset.official;link.target='_blank';link.rel='noopener';link.textContent=' 查看原始数据集';$('family-data-source').appendChild(link);}
      $('family-gallery').innerHTML=rows.map(s=>`<label class="family-sample"><input class="family-sample-check" type="checkbox" data-split="${esc(s.split)}" value="${esc(s.key)}"><a href="${esc(s.image_url)}" target="_blank" rel="noopener" aria-label="查看裁图大图"><img src="${esc(s.image_url)}" alt="${esc(s.category_name)}候选裁图" loading="lazy"></a><strong>${esc(s.category_name)}</strong><span>${esc(s.source_category?.name||'')} · 原图 ${esc(s.source_image_id)}</span><span>${esc(s.split)} · ${labels[s.appearance_status||'unreviewed']}</span><span>质量标签：来源未提供</span>${s.evaluation_reserved?'<span>已封存评价，不能加入参考</span>':''}</label>`).join('');
      $('family-gallery-status').textContent=d.count?`第 ${offset+1}—${offset+rows.length} 张，共 ${d.count} 张`:'暂无候选样本';$('family-prev').disabled=offset===0;$('family-next').disabled=offset+rows.length>=d.count;
    }catch(e){if(ticket===sequence)$('family-gallery-status').textContent=e.message;}
  }
  async function review(status){
    const key=current,ids=[...$('family-gallery').querySelectorAll('.family-sample-check:checked')].map(e=>e.value);
    if(!ids.length){$('family-gallery-status').textContent='请先勾选图片';return;}
    const payload={sample_keys:ids,appearance_status:status,reviewer:$('family-reviewer').value,note:$('family-review-note').value};
    const buttons=[...$('family-appearance').querySelectorAll('button')];buttons.forEach(b=>b.disabled=true);
    try{const d=await api(`/api/product-families/${encodeURIComponent(key)}/appearance/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(key!==current)return;await load();await gallery(false);$('family-gallery-status').textContent=d.note;}
    catch(e){if(key===current)$('family-gallery-status').textContent=e.message;}finally{buttons.forEach(b=>b.disabled=false);}
  }
  return {init,load,gallery,selected:()=>current};
})();
