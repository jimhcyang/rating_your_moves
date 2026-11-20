(() => {
  const views = ['study', 'example'];
  const frame = document.getElementById('demo-frame');
  function showView() {
    const current = location.hash === '#study' ? 'study' : 'example';
    views.forEach(name => {
      document.getElementById(name).hidden = name !== current;
      const tab = document.getElementById(`${name}-tab`);
      if (name === current) tab.setAttribute('aria-current', 'page');
      else tab.removeAttribute('aria-current');
    });
    document.title = `${current === 'study' ? 'The Study' : 'Rating Your Moves'} · Jim Yang`;
    if (current === 'example') requestAnimationFrame(resizeDemo);
  }
  function resizeDemo() {
    if (document.getElementById('example').hidden) return;
    try {
      const height = frame.contentDocument.body.scrollHeight;
      if (height > 100) frame.style.height = `${Math.ceil(height) + 2}px`;
    } catch (_) { /* The standalone example remains accessible if origins differ. */ }
  }
  window.addEventListener('message', event => {
    if (event.source !== frame.contentWindow || event.origin !== location.origin) return;
    if (event.data?.type === 'rym-demo-height' && Number.isFinite(event.data.height)) resizeDemo();
  });
  frame.addEventListener('load', resizeDemo);
  window.addEventListener('resize', resizeDemo);
  window.addEventListener('hashchange', () => {
    showView();
    window.scrollTo({ top: 0, behavior: 'instant' });
  });
  const replay = document.querySelector('.replay-detail');
  replay.addEventListener('toggle', () => {
    const img = replay.querySelector('img');
    if (replay.open) img.src = img.dataset.src;
    else img.removeAttribute('src');
  });
  showView();
})();
