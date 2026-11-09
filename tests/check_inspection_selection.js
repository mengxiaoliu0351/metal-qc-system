const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Body{constructor(){this.values={};}append(key,value){this.values[key]=value;}}
const sandbox={window:{},FormData:Body};vm.createContext(sandbox);vm.runInContext(fs.readFileSync('src/web/static/inspection-selection.js','utf8'),sandbox);
const api=sandbox.window.InspectionSelection,file={name:'part.png'},other={name:'other.png'};
const family={key:'screw',mode:'family',family_key:'screw',batch_key:'family:screw',ready:true};
let request=api.request(family,file,null);assert.equal(request.url,'/api/product-families/screw/inspection/detect');assert.equal(request.options.body.values.file,file);assert.equal(request.options.body.values.product,undefined);
request=api.request(family,file,{id:'fixed',file});assert.equal(request.url,'/api/product-families/screw/inspection/examples/fixed/detect');assert.equal(request.options.body,undefined);
request=api.request(family,other,{id:'fixed',file});assert.equal(request.url,'/api/product-families/screw/inspection/detect','stale demo identity must not override new uploaded image');
request=api.request({key:'metal_nut',mode:'product',product_key:'metal_nut'},file,{id:'fixed',file});assert.equal(request.url,'/api/detect');assert.equal(request.options.body.values.product,'metal_nut');
assert.throws(()=>api.request({...family,ready:false},file,null));assert.throws(()=>api.request(null,file,null));
const app=fs.readFileSync('src/web/static/app.js','utf8'),familyUI=fs.readFileSync('src/web/static/family-ui.js','utf8');
assert.doesNotMatch(app,/pages\.family\s*=/);assert.doesNotMatch(app,/\['detect','family','batch'/);assert.match(app,/if\(key==='family'\)\{key='detect'/);
assert.match(app,/window\.InspectionSelection\.request\(currentTarget\(\)/);assert.match(app,/value="\$\{esc\(t\.batch_key\)\}"/);assert.doesNotMatch(familyUI,/page-family|FamilyInspectionUI.*init/);
const html=fs.readFileSync('src/web/static/index.html','utf8'),productsUI=fs.readFileSync('src/web/static/product-ui.js','utf8');
assert.doesNotMatch(html,/screw-showcase-ui\.js|screw-showcase\.css/,'ordinary pages must not load a separate screw showcase');
assert.doesNotMatch(app,/ScrewShowcaseUI/);assert.match(productsUI,/研发设置/);
assert.match(familyUI,/showcase\/export/,'detailed evaluation remains in folded research settings');
assert.match(app,/查看参考与匹配说明<\/summary>/,'technical matching details stay folded');
console.log('One object selector verified: family dispatch, recipe hiding, old-link redirect, batch identity and stale-demo protection.');
