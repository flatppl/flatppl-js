// Tree-shaken echarts build: echarts/core + only the chart types and
// components the FlatPPL visualizer uses (audited from packages/viewer/src).
// Built to an IIFE (globalName 'echarts') so the webview's <script> load and
// the viewer's `declare const echarts` are unchanged. Canvas renderer only
// (the webview CSP forbids eval; the SVG renderer is unused).
//
// IF YOU ADD A NEW CHART TYPE OR A NEW top-level option component in the
// viewer, add it to the `use([...])` list below — a missing registration
// makes that chart/feature silently no-op. test/echarts-entry.test.js guards
// the chart-type set against drift.
import * as echarts from 'echarts/core';
import { LineChart, BarChart, ScatterChart, HeatmapChart, CustomChart } from 'echarts/charts';
import {
  GridComponent, TitleComponent, LegendComponent, TooltipComponent,
  ToolboxComponent, DataZoomComponent, DataZoomInsideComponent,
  VisualMapContinuousComponent, AxisPointerComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  LineChart, BarChart, ScatterChart, HeatmapChart, CustomChart,
  GridComponent, TitleComponent, LegendComponent, TooltipComponent,
  ToolboxComponent, DataZoomComponent, DataZoomInsideComponent,
  VisualMapContinuousComponent, AxisPointerComponent,
  CanvasRenderer,
]);

export * from 'echarts/core';
