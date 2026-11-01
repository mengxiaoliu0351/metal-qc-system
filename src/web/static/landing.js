'use strict';

// The public introduction and the inspection station are separate views.
window.MetalQCLanding = {
  html: `
  <section id="landing-screen" class="landing-screen" aria-label="MetalQC 系统首页">
    <div class="landing-atmosphere" aria-hidden="true"><div class="landing-beam"></div><div class="landing-stars"></div></div>
    <header class="landing-header">
      <a class="landing-logo" href="#home" aria-label="MetalQC 系统首页"><span class="landing-monogram">M<span>Q</span></span><span>MetalQC<small>金属表面智能质检</small></span></a>
      <nav class="landing-nav" aria-label="首页导航"><a href="#capabilities">系统能力</a><a href="#workflow">应用流程</a><a class="landing-nav-enter" href="#detect">进入工作台 <span aria-hidden="true">↗</span></a></nav>
    </header>
    <main id="landing-main" class="landing-main" tabindex="-1">
      <section class="landing-hero" aria-labelledby="landing-title">
        <div class="landing-part part-left" aria-hidden="true"><img src="/static/landing/machined-part.svg" alt=""><span class="part-coordinate">SURFACE / PERCEPTION</span></div>
        <div class="landing-part part-right" aria-hidden="true"><img src="/static/landing/machined-part.svg" alt=""><span class="part-coordinate">VISION / REVIEW</span></div>
        <span class="landing-callout callout-left" aria-hidden="true"><i></i>局部异常定位<span>ANOMALY LOCALIZATION</span></span>
        <span class="landing-callout callout-right" aria-hidden="true"><i></i>视觉复核 × 知识检索<span>VISUAL REVIEW & RETRIEVAL</span></span>
        <div class="landing-hero-copy">
          <p class="landing-kicker"><span></span>面向金属表面缺陷的双链路智能质检</p>
          <h1 id="landing-title">洞察金属表面<br><span>让缺陷有迹可循</span></h1>
          <p class="landing-intro">从局部异常定位，到视觉复核与知识检索。<br>常规样本自动处理，例外与抽查交给人工核验。</p>
          <div class="landing-actions"><a class="landing-cta" href="#detect"><span>进入检测工作台</span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12h16m-6-6 6 6-6 6"/></svg></a><a class="landing-secondary" href="#capabilities">了解系统能力 <span aria-hidden="true">↓</span></a></div>
          <p class="landing-cta-note">工件检测 · 人工复核 · 模型迭代 · AI 助手</p>
        </div>
        <div class="landing-hero-base"><span class="landing-micro">METAL SURFACE INSPECTION</span><div class="landing-products"><span>金属螺母</span><i></i><span>金属螺丝</span><i></i><span>金属网格</span></div><a href="#capabilities" class="landing-scroll" aria-label="向下了解系统">SCROLL TO EXPLORE <span aria-hidden="true">↓</span></a></div>
      </section>
      <section id="capabilities" class="landing-capabilities" aria-labelledby="capabilities-title">
        <div class="landing-section-heading"><div><span class="landing-section-index">01 / CAPABILITIES</span><h2 id="capabilities-title">从看见异常，到辅助判断。</h2></div><p>将本地感知与云端认知接入同一条检验流程。</p></div>
        <div class="landing-evidence-layout">
          <div class="landing-evidence"><div class="evidence-bar"><span>金属螺母 · 表面划痕示例</span><small>实际样品与推理输出</small></div><div class="evidence-pair"><figure><img src="/static/landing/nut-original.png" alt="MVTec 螺母表面划痕样品原图" loading="lazy" width="1024" height="1024"><figcaption><span>A</span> 工件原图</figcaption></figure><figure><img src="/static/landing/nut-response.png" alt="同一螺母图像的局部异常热力图" loading="lazy" width="1024" height="1024"><figcaption><span>B</span> 局部异常响应</figcaption></figure></div><p>热力图用于辅助定位，质量结论需结合原图、适用标准与人工复核。</p></div>
          <div class="landing-capability-list"><article><span>01</span><div><h3>本地感知 · 定位异常</h3><p>按产品加载检测配置，输出整图评分、局部热力图与缺陷类型线索。</p></div></article><article><span>02</span><div><h3>灰区复核 · 补充依据</h3><p>对中间风险样品调用视觉模型，结合知识库检索片段辅助判断。</p></div></article><article><span>03</span><div><h3>记录追溯 · 辅助复核</h3><p>回看检验记录、导出 PDF 报告，也可携带检测图像向助手继续提问。</p></div></article><a href="#detect" class="landing-inline-link">打开检验工位 <span aria-hidden="true">↗</span></a></div>
        </div>
      </section>
      <section id="workflow" class="landing-workflow" aria-labelledby="workflow-title"><div class="landing-section-heading"><div><span class="landing-section-index">02 / WORKFLOW</span><h2 id="workflow-title">让技术落在实际操作里。</h2></div><a href="#detect" class="landing-inline-link">开始一次检验 <span aria-hidden="true">↗</span></a></div><ol class="landing-process"><li><span>01</span><h3>选产品，载入图像</h3><p>选择对应工件，也可用内置真实试件演示。</p></li><li><span>02</span><h3>查看检测与风险分流</h3><p>并排观察原图与热图，灰区补充视觉分析。</p></li><li><span>03</span><h3>核对依据，整理记录</h3><p>结合分析建议复核，导出报告供后续归档。</p></li></ol></section>
    </main>
    <footer class="landing-footer"><span>MetalQC <b>/</b> 双链路智能质检辅助系统</span><span>AI 智能分流 · 人工例外复核</span><a href="#detect">进入工作台 ↗</a></footer>
  </section>`
};
