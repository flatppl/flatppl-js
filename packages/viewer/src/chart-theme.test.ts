// @ts-nocheck — Node drives real ECharts SVG output; no browser or canvas needed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as echarts from 'echarts';
import { observeChartTheme } from './chart-theme.ts';

test('host theme changes recolor mounted charts without changing data, zoom or selection, and disconnect on dispose', (t) => {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 500, height: 300 });
  t.after(() => chart.dispose());
  chart.setOption({
    animation: false,
    title: { text: 'Samples', textStyle: { color: '#333' } },
    legend: { selected: { hidden: false } },
    xAxis: { name: 'x' }, yAxis: { name: 'density' },
    dataZoom: [{ type: 'inside', start: 20, end: 80 }],
    tooltip: {}, toolbox: { feature: { dataZoom: {} } },
    series: [
      { name: 'visible', type: 'line', data: [[0, 1], [1, 3], [2, 2]], itemStyle: { color: '#82b366' } },
      { name: 'hidden', type: 'line', data: [[0, 2], [1, 4], [2, 1]] },
      { type: 'lines', coordinateSystem: 'cartesian2d', data: [{ coords: [[0, 0], [1, 1]] }], lineStyle: { color: '#333' } },
    ],
  });
  const baseline = chart.getOption();
  const matrix = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 300, height: 300 });
  t.after(() => matrix.dispose());
  matrix.setOption({
    animation: false,
    xAxis: { type: 'category', data: ['1'] }, yAxis: { type: 'category', data: ['1'] },
    visualMap: { min: 0, max: 10, inRange: { color: ['#c6dbef', '#08306b'] } },
    series: [{ type: 'heatmap', data: [[0, 0, 7]], label: { show: true, color: '#333' } }],
  });
  const matrixBaseline = matrix.getOption();
  const body = {}, html = {}, chartElement = {}, matrixElement = {};
  const root = { ownerDocument: { body, documentElement: html }, querySelectorAll: () => [chartElement, matrixElement] };
  let foreground = '#333', background = '#fff', callback, connected = false;
  const observed = [];
  class Observer {
    constructor(fn) { callback = fn; connected = true; }
    observe(node, options) { observed.push({ node, options }); }
    disconnect() { connected = false; }
  }
  const old = Object.getOwnPropertyDescriptors(globalThis);
  globalThis.MutationObserver = Observer;
  globalThis.getComputedStyle = () => ({ color: foreground, getPropertyValue: () => background });
  globalThis.echarts = { getInstanceByDom: (node) => node === chartElement ? chart : node === matrixElement ? matrix : null };
  t.after(() => {
    for (const key of ['MutationObserver', 'getComputedStyle', 'echarts']) {
      if (old[key]) Object.defineProperty(globalThis, key, old[key]); else delete globalThis[key];
    }
  });
  const stop = observeChartTheme(root);
  assert.deepEqual(observed.map((entry) => entry.node), [html, body]);
  assert.deepEqual(observed[0].options.attributeFilter, ['class', 'style', 'data-theme']);
  for (const [color, bg] of [['#eee', '#222'], ['#333', '#fff']]) {
    foreground = color; background = bg;
    if (connected) callback();
    const option = chart.getOption();
    assert.equal(option.title[0].textStyle.color, color);
    assert.equal(option.xAxis[0].axisLabel.color, color);
    assert.equal(option.yAxis[0].nameTextStyle.color, color);
    assert.equal(option.legend[0].textStyle.color, color);
    assert.equal(option.toolbox[0].iconStyle.borderColor, color);
    assert.equal(option.tooltip[0].backgroundColor, bg);
    assert.equal(option.series[2].lineStyle.color, color);
    assert.equal(option.series[0].itemStyle.color, '#82b366');
    assert.deepEqual(option.series.map((series) => series.data), baseline.series.map((series) => series.data));
    assert.deepEqual(option.dataZoom, baseline.dataZoom);
    assert.deepEqual(option.legend[0].selected, baseline.legend[0].selected);
    const matrixOption = matrix.getOption();
    assert.equal(matrixOption.series[0].label.color, color);
    assert.deepEqual(matrixOption.series[0].data, matrixBaseline.series[0].data);
    assert.deepEqual(matrixOption.visualMap[0].inRange, matrixBaseline.visualMap[0].inRange);
    assert.equal(matrixOption.visualMap[0].min, 0);
    assert.equal(matrixOption.visualMap[0].max, 10);
  }
  stop();
  foreground = '#eee';
  if (connected) callback();
  assert.equal(chart.getOption().title[0].textStyle.color, '#333');
  assert.equal(matrix.getOption().series[0].label.color, '#333');
});
