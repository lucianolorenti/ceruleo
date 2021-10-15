import React, { useEffect, useState } from "react";
import { API } from "./API";

import { ViolinPlot, BoxPlot } from '@visx/stats';
import { Group } from '@visx/group';
import { LinearGradient } from '@visx/gradient';
import { scaleBand, scaleLinear } from '@visx/scale';
import { Stats } from '@visx/mock-data/lib/generators/genStats';
import { getSeededRandom, getRandomNormal } from '@visx/mock-data';
import { withTooltip, Tooltip, defaultStyles as defaultTooltipStyles } from '@visx/tooltip';
import { WithTooltipProvidedProps } from '@visx/tooltip/lib/enhancers/withTooltip';
import { PatternLines } from '@visx/pattern';
import { CircularProgress, Grid } from "@mui/material";
import FeatureSelector from "./FeatureSelector";

interface BasicDurationProps {
    api: API
}

interface TooltipData {
    name?: string;
    min?: number;
    median?: number;
    max?: number;
    firstQuartile?: number;
    thirdQuartile?: number;
}

export type StatsPlotProps = {
    width: number;
    height: number;
    api: API
};



const x = (d: Stats) => d.boxPlot.x;
const min = (d: Stats) => d.boxPlot.min;
const max = (d: Stats) => d.boxPlot.max;
const median = (d: Stats) => d.boxPlot.median;
const firstQuartile = (d: Stats) => d.boxPlot.firstQuartile;
const thirdQuartile = (d: Stats) => d.boxPlot.thirdQuartile;
const outliers = (d: Stats) => d.boxPlot.outliers;

const DurationBoxPlot = withTooltip<StatsPlotProps, TooltipData>(
    (props: StatsPlotProps & WithTooltipProvidedProps<TooltipData>) => {
        const [boxPlot, setBoxPlot] = useState<Array<Stats>>(null)
        
        useEffect(() => {
            props.api.durationDistribution(setBoxPlot)
        }, [])
       
        if (boxPlot == null) {
            return <CircularProgress />;
        }
        console.log(boxPlot)
        // bounds
        const xMax = props.width;
        const yMax = props.height - 120;
        const data = boxPlot
        // scales
        const xScale = scaleBand<string>({
            range: [0, xMax],
            round: true,
            domain: data.map(x),
            padding: 0.4,
        });

        const values = data.reduce((allValues, { boxPlot }) => {
            allValues.push(boxPlot.min, boxPlot.max);
            return allValues;
        }, [] as number[]);
        const minYValue = Math.min(...values);
        const maxYValue = Math.max(...values);

        const yScale = scaleLinear<number>({
            range: [yMax, 0],
            round: true,
            domain: [minYValue, maxYValue],
        });

        const boxWidth = xScale.bandwidth();
        const constrainedWidth = Math.min(40, boxWidth);

        return props.width < 10 ? null : (
            <div style={{ position: 'relative' }}>
                <svg width={props.width} height={props.height}>
                    <LinearGradient id="statsplot" to="#8b6ce7" from="#87f2d4" />
                    <rect x={0} y={0} width={props.width} height={props.height} fill="url(#statsplot)" rx={14} />
                    <PatternLines
                        id="hViolinLines"
                        height={3}
                        width={3}
                        stroke="#ced4da"
                        strokeWidth={1}
                        // fill="rgba(0,0,0,0.3)"
                        orientation={['horizontal']}
                    />
                    <Group top={40}>
                        {data.map((d: Stats, i) => (
                            <g key={i}>
                                <ViolinPlot
                                    data={d.binData}
                                    stroke="#dee2e6"
                                    left={xScale(x(d))!}
                                    width={constrainedWidth}
                                    valueScale={yScale}
                                    fill="url(#hViolinLines)"
                                />
                                <BoxPlot
                                    min={min(d)}
                                    max={max(d)}
                                    left={xScale(x(d))! + 0.3 * constrainedWidth}
                                    firstQuartile={firstQuartile(d)}
                                    thirdQuartile={thirdQuartile(d)}
                                    median={median(d)}
                                    boxWidth={constrainedWidth * 0.4}
                                    fill="#FFFFFF"
                                    fillOpacity={0.3}
                                    stroke="#FFFFFF"
                                    strokeWidth={2}
                                    valueScale={yScale}
                                    outliers={outliers(d)}
                                    minProps={{
                                        onMouseOver: () => {
                                            props.showTooltip({
                                                tooltipTop: yScale(min(d)) ?? 0 + 40,
                                                tooltipLeft: xScale(x(d))! + constrainedWidth + 5,
                                                tooltipData: {
                                                    min: min(d),
                                                    name: x(d),
                                                },
                                            });
                                        },
                                        onMouseLeave: () => {
                                            props.hideTooltip();
                                        },
                                    }}
                                    maxProps={{
                                        onMouseOver: () => {
                                            props.showTooltip({
                                                tooltipTop: yScale(max(d)) ?? 0 + 40,
                                                tooltipLeft: xScale(x(d))! + constrainedWidth + 5,
                                                tooltipData: {
                                                    max: max(d),
                                                    name: x(d),
                                                },
                                            });
                                        },
                                        onMouseLeave: () => {
                                            props.hideTooltip();
                                        },
                                    }}
                                    boxProps={{
                                        onMouseOver: () => {
                                            props.showTooltip({
                                                tooltipTop: yScale(median(d)) ?? 0 + 40,
                                                tooltipLeft: xScale(x(d))! + constrainedWidth + 5,
                                                tooltipData: {
                                                    ...d.boxPlot,
                                                    name: x(d),
                                                },
                                            });
                                        },
                                        onMouseLeave: () => {
                                            props.hideTooltip();
                                        },
                                    }}
                                    medianProps={{
                                        style: {
                                            stroke: 'white',
                                        },
                                        onMouseOver: () => {
                                            props.showTooltip({
                                                tooltipTop: yScale(median(d)) ?? 0 + 40,
                                                tooltipLeft: xScale(x(d))! + constrainedWidth + 5,
                                                tooltipData: {
                                                    median: median(d),
                                                    name: x(d),
                                                },
                                            });
                                        },
                                        onMouseLeave: () => {
                                            props.hideTooltip();
                                        },
                                    }}
                                />
                            </g>
                        ))}
                    </Group>
                </svg>

                {props.tooltipOpen && props.tooltipData && (
                    <Tooltip
                        top={props.tooltipTop}
                        left={props.tooltipLeft}
                        style={{ ...defaultTooltipStyles, backgroundColor: '#283238', color: 'white' }}
                    >
                        <div>
                            <strong>{props.tooltipData.name}</strong>
                        </div>
                        <div style={{ marginTop: '5px', fontSize: '12px' }}>
                            {props.tooltipData.max && <div>max: {props.tooltipData.max}</div>}
                            {props.tooltipData.thirdQuartile && <div>third quartile: {props.tooltipData.thirdQuartile}</div>}
                            {props.tooltipData.median && <div>median: {props.tooltipData.median}</div>}
                            {props.tooltipData.firstQuartile && <div>first quartile: {props.tooltipData.firstQuartile}</div>}
                            {props.tooltipData.min && <div>min: {props.tooltipData.min}</div>}
                        </div>
                    </Tooltip>
                )}
            </div>
        );
    },
);

interface DurationProps {
    api: API
}

export default function Duration(props: DurationProps) {
    const [categoricalFeature, setCategoricalFeature] = useState<string[]>([''])
   
    return (

        <Grid container spacing={2} style={{ marginTop: '1em' }}>

        <Grid item xs={12}>
             <FeatureSelector api={props.api} setCurrentFeatures={setCategoricalFeature} features={categoricalFeature} />
        </Grid>
        <Grid item xs={12}>
            <DurationBoxPlot api={props.api} width={600} height={400} />
        </Grid>

    </Grid>

       
    )
}