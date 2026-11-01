(() => {
  const apply = () => {
    const footer = document.querySelector('.sidebar-footer');
    if (footer) {
      const form = document.createElement('form');
      form.method = 'post'; form.action = '/logout';
      const button = document.createElement('button');
      button.type = 'submit'; button.textContent = '退出演示';
      button.style.cssText = 'margin-left:12px;padding:7px 10px;border:1px solid #a1bbd8;border-radius:4px;background:transparent;color:#fff;font-size:13px;cursor:pointer';
      form.appendChild(button); footer.appendChild(form); footer.style.display = 'flex';
    }
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', apply); else apply();
})();
