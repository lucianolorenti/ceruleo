import React from "react";
import styles from './ScoreBar.scss'
import AutoSizer from 'react-virtualized-auto-sizer';
import { Tooltip } from "@mui/material";

import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { round } from "reliable-round";
// Import charts, all with Chart suffix
import {

  BarChart,

} from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
  DatasetComponent,
  LegendComponent,
} from 'echarts/components';
// Import renderer, note that introducing the CanvasRenderer or SVGRenderer is a required step
import {
  CanvasRenderer,
} from 'echarts/renderers';

// Register the required components
echarts.use(
  [TitleComponent, LegendComponent, TooltipComponent, GridComponent, BarChart, CanvasRenderer]
);

interface ScoreBarProps {
  max: number
  min: number
  mean: number
}

const ScoreBar = (props: ScoreBarProps) => {

  const bgcolor = 'red'
  const max_percentage = round(props.max, 2)
  const min_percentage = round(props.min, 2)
  const mean_percentage = round(props.mean, 2)

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        // Use axis to trigger tooltip
        type: 'shadow' // 'shadow' as default; can also be 'line' or 'shadow'
      }
    },
    grid: {
      top: '0%',
      left: '0%',
      right: '0%',
      bottom: '0%',
      height: '20px',
      containLabel: false
    },
    xAxis: {
      type: 'value',
      show: false,
      min: 0,
      max: 100
    },
    yAxis: {
      type: 'category',
      data: ['Nulla']
    },
    series: [
      {
        name: 'Min',
        type: 'bar',
        stack: 'total',
        label: {
          show: true
        },
        emphasis: {
          focus: 'series'
        },
        data: [min_percentage]
      },
      {
        name: 'Mean',
        type: 'bar',
        stack: 'total',
        label: {
          show: true
        },
        emphasis: {
          focus: 'series'
        },
        data: [mean_percentage,]
      },
      {
        name: 'Max',
        type: 'bar',
        stack: 'total',
        label: {
          show: true
        },
        emphasis: {
          focus: 'series'
        },
        data: [max_percentage]
      }
    ]
  };

  return (

    <div className={styles.containerStyles}>


      <ReactEChartsCore
        echarts={echarts}
        option={option}

      />






    </div>

  );
};

export default ScoreBar;
