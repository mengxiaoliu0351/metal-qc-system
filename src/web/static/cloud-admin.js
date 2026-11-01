(() => {
  const explain = () => {
    const form = document.getElementById('iteration-form');
    if (!form) return;
    const notice = document.createElement('p');
    notice.className = 'panel-footnote';
    notice.textContent = '云端演示配置：人工复核正常记录；候选训练与阈值评估请在本机完成后迁移。此实例暂不发起训练任务。';
    form.after(notice);
    const start = document.getElementById('iteration-start');
    start.disabled = true;
    start.textContent = '训练请在本机运行';
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', explain);
  else explain();
})();
