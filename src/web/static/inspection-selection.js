'use strict';
// The visible object and the internal recipe are different identities.
window.InspectionSelection={
  targets(data){
    const targets=[...(data.inspection_targets||[])];
    if(data.fallback)targets.push({key:data.fallback.key,name:'其他／未建库工件 · AI辅助',mode:'general',product_key:data.fallback.key,ready:true});
    return targets;
  },
  find(targets,key){return targets.find(t=>t.key===key);},
  request(target,file,exampleId){
    if(!target||target.ready===false)throw new Error('该检测对象尚未准备好');
    const example=exampleId&&typeof exampleId==='object'&&exampleId.file===file?exampleId.id:null;
    if(target.mode==='family'&&example)return {url:`/api/product-families/${encodeURIComponent(target.family_key)}/inspection/examples/${encodeURIComponent(example)}/detect`,options:{method:'POST'}};
    const body=new FormData();body.append('file',file);
    if(target.mode==='family')return {url:`/api/product-families/${encodeURIComponent(target.family_key)}/inspection/detect`,options:{method:'POST',body}};
    body.append('product',target.product_key);return {url:'/api/detect',options:{method:'POST',body}};
  }
};
