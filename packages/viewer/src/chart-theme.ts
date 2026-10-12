// Refresh mounted ECharts presentation after a host theme change. Patching
// component styles keeps sampled data, selected legends and dataZoom intact.
// Web themes use data-theme on <html>; VS Code changes body classes/styles.

/** Update only the chart components whose colors come from the host theme. */
export function updateChartTheme(chart: any, foreground: string, background: string): void {
  const model = chart.getModel();
  const patch: Record<string, any> = { textStyle: { color: foreground } };
  const components: Record<string, any> = {
    title: { textStyle: { color: foreground }, subtextStyle: { color: foreground } },
    legend: { textStyle: { color: foreground }, pageTextStyle: { color: foreground } },
    visualMap: { textStyle: { color: foreground } },
    toolbox: { iconStyle: { borderColor: foreground } },
    tooltip: { textStyle: { color: foreground }, backgroundColor: background },
    xAxis: {
      nameTextStyle: { color: foreground }, axisLabel: { color: foreground },
      axisLine: { lineStyle: { color: foreground } },
      axisTick: { lineStyle: { color: foreground } },
      splitLine: { lineStyle: { color: foreground } },
    },
  };
  components.yAxis = components.xAxis;
  for (const [name, style] of Object.entries(components)) {
    const entries: any[] = [];
    model.eachComponent(name, (component: any) => { entries.push({ id: component.id, ...style }); });
    if (entries.length) patch[name] = entries;
  }
  // Contours and heatmap value labels use foreground ink. Bars, points and heatmap
  // ramps encode data and must keep their own colors. Read component metadata
  // instead of getOption(), which clones every sample array in a large plot.
  const seriesStyles: any[] = [];
  model.eachSeries((series: any) => {
    if (series.subType === 'lines') seriesStyles.push({ id: series.id, lineStyle: { color: foreground } });
    if (series.subType === 'heatmap') seriesStyles.push({ id: series.id, label: { color: foreground } });
  });
  if (seriesStyles.length) patch.series = seriesStyles;
  chart.setOption(patch, { silent: true });
}

/** Observe host theme attributes for this mount. Return its teardown hook. */
export function observeChartTheme(root: HTMLElement): () => void {
  const doc = root.ownerDocument;
  let previous = '';
  const refresh = () => {
    const style = getComputedStyle(doc.body);
    const foreground = style.color || '#ccc';
    const background = style.getPropertyValue('--vscode-editorWidget-background').trim()
      || style.backgroundColor;
    const key = foreground + '\n' + background;
    if (key === previous) return;
    previous = key;
    for (const element of [root, ...Array.from(root.querySelectorAll('div'))]) {
      const chart = echarts.getInstanceByDom(element);
      if (chart && !chart.isDisposed()) updateChartTheme(chart, foreground, background);
    }
  };
  if (typeof MutationObserver !== 'function') return () => {};
  const observer = new MutationObserver(refresh);
  const attrs = { attributes: true, attributeFilter: ['class', 'style', 'data-theme'] };
  observer.observe(doc.documentElement, attrs);
  observer.observe(doc.body, attrs);
  refresh();
  return () => observer.disconnect();
}
