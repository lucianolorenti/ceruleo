import { LineSvgProps, ResponsiveLine } from '@nivo/line'
import React, { useEffect, useState } from "react";
import { LineData } from './API';

interface LinePlotsProps {
    data: Array<LineData>
}



export default function LinePlot(props:LinePlotsProps) {
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