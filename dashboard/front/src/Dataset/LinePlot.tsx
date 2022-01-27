import { Tooltip } from '@mui/material';
import { LineSvgProps, ResponsiveLine } from '@nivo/line'
import React, { useEffect, useState } from "react";
import { VictoryAxis, VictoryChart, VictoryLegend, VictoryLine, VictoryTheme } from 'victory';
import { LineData } from './API';

interface LinePlotsProps {
    data: Array<LineData>
}




export default function LinePlot(props: LinePlotsProps) {
    const maxima = props.data.map(
        (dataset) => Math.max(...dataset.data.map((d) => d.y))
    );
    const maximaX = props.data.map(
        (dataset) => Math.max(...dataset.data.map((d) => d.x))
    );
    const tickPadding = [0, -15];
    const anchors = ["end", "start"];
    const colors = ["black", "red", "blue"];
    const offsetX = [40, 100]

    return (<VictoryChart

        height={100}
        padding={{ left: 45, bottom: 45, top: 15 }}
        domain={{ y: [0, 1] }}

    >
        <VictoryLegend x={125} y={50}
            title="Legend"
            centerTitle
            orientation="horizontal"
            style={{ border: { stroke: "black" }, title: { fontSize: 2 } }}
            data={props.data.map((d: LineData, i: number) => {
                return {
                    'name': d.id,
                    'fill': colors[i]
                }
            })}
        />
        <VictoryAxis style={{
            tickLabels: { fontSize: 3 }
        }} />
        {props.data.map((d, i) => (
            <VictoryAxis dependentAxis
                orientation={(i == 0 ? 'left' : 'right')}
                key={i}
                style={{
                    axis: { stroke: colors[i] },
                    ticks: { padding: tickPadding[i] },
                    tickLabels: { fontSize: 3, fill: colors[i], textAnchor: anchors[i] }


                }}
                // Use normalized tickValues (0 - 1)
                tickValues={[0.25, 0.5, 0.75, 1]}
                // Re-scale ticks by multiplying by correct maxima
                tickFormat={(t) => (t * maxima[i]).toFixed(2)}

            />
        ))}

        {props.data.map((l: LineData, i) => (
            <VictoryLine
                key={i}
                data={l.data}
                style={{ data: { stroke: colors[i], strokeWidth: 0.1 } }}
                y={(datum) => datum.y / maxima[i]}

            />
        ))}
    </VictoryChart>)

}

export function LinePlot1(props: LinePlotsProps) {
    return (
        <div style={{ height: '400px' }} >
            <ResponsiveLine

                data={props.data}
                margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
                xScale={{ type: 'linear', max: 'auto', min: 'auto' }}
                yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false }}
                yFormat=" >-.2f"
                curve='monotoneX'
                enablePoints={false}
                axisTop={null}
                axisRight={null}
                axisBottom={{

                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'transportation',
                    legendOffset: 36,
                    legendPosition: 'middle'
                }}
                axisLeft={{

                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'count',
                    legendOffset: -40,
                    legendPosition: 'middle'
                }}
                pointSize={10}
                pointColor={{ theme: 'background' }}
                pointBorderWidth={2}
                pointBorderColor={{ from: 'serieColor' }}
                pointLabelYOffset={-12}
                useMesh={true}
                legends={[
                    {
                        anchor: 'bottom-right',
                        direction: 'column',
                        justify: false,
                        translateX: 100,
                        translateY: 0,
                        itemsSpacing: 0,
                        itemDirection: 'left-to-right',
                        itemWidth: 80,
                        itemHeight: 20,
                        itemOpacity: 0.75,
                        symbolSize: 12,
                        symbolShape: 'circle',
                        symbolBorderColor: 'rgba(0, 0, 0, .5)',
                        effects: [
                            {
                                on: 'hover',
                                style: {
                                    itemBackground: 'rgba(0, 0, 0, .03)',
                                    itemOpacity: 1
                                }
                            }
                        ]
                    }
                ]}

            />
        </div>)

}