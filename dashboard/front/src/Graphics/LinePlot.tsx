
import React, { useEffect, useState } from "react";

import { LineData, PlotData } from '../Dataset/Network/Responses';
import ReactApexChart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';

interface LinePlotsProps {
    feature: string;
    series: PlotData[];
    height: number;
}


function build_options(title: string): ApexOptions {
    return {
        chart: {
            height: 350,
            type: "line",
            stacked: false,
            animations: {
                enabled: false
            }
        },
        dataLabels: {
            enabled: false
        },

        title: {
            text: title,
            align: 'left',
            offsetX: 110
        },
        markers: {
            size: 0
        },
        yaxis: [
            {
                axisTicks: {
                    show: true,
                },
                axisBorder: {
                    show: true,
                    color: '#008FFB'
                },
                labels: {
                    style: {
                        colors: '#008FFB',
                    }
                },
                tooltip: {
                    enabled: true
                }
            }

        ],
        xaxis: {
            tickAmount: 20
        },
        tooltip: {
            fixed: {
                enabled: true,
                position: 'topLeft', // topRight, topLeft, bottomRight, bottomLeft
                offsetY: 30,
                offsetX: 60
            },
        },
        legend: {
            horizontalAlign: 'left',
            offsetX: 40
        }
    }
}



export default function LinePlot(props: LinePlotsProps) {
    return (<ReactApexChart
        options={build_options(props.feature)}
        series={props.series}
        type="line"
        height={350} />
    )
}