/* Public entry: management remains available through the private SSH-forwarded entry. */
(() => {
  const style = document.createElement('style');
  style.textContent = '#review-button,.feedback-actions,[data-page="review"],[data-page="history"],[data-page="iterations"],[data-page="dashboard"]{display:none!important}';
  document.head.appendChild(style);
  const management = new Set(['review', 'history', 'iterations', 'dashboard']);
  const apply = () => {
    document.querySelectorAll('[data-page]').forEach(node => {
      if (management.has(node.dataset.page)) node.hidden = true;
    });
    document.querySelectorAll('#review-button,.feedback-actions').forEach(node => { node.hidden = true; });
    if (!document.getElementById('public-demo-notice')) {
      const notice = document.createElement('div');
      notice.id = 'public-demo-notice';
      notice.style.cssText = 'padding:10px 18px;background:#13282d;color:#d8e8e8;font:12px/1.7 sans-serif;text-align:center;position:relative;z-index:20';
      notice.textContent = '公开演示 · 支持已建库的螺母、螺丝、网格；图片可能发送至云端AI，请勿上传敏感图片。每个网络地址每日限20次检测或咨询，报告请在当前浏览器1小时内下载。人工复核及模型管理由管理员操作。';
      document.body.prepend(notice);
    }
    if (management.has(location.hash.slice(1))) location.hash = 'detect';
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', apply);
  else apply();
  window.addEventListener('hashchange', apply);
})();
