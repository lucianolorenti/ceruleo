
import { CircularProgress } from "@mui/material";

import React, { ReactNode, useEffect, useState } from "react";
import { API, BoxPlotData } from "./API";


import ReactApexChart from "react-apexcharts";
import { ApexOptions } from "apexcharts";

interface PropsSamplingRate {
    api: API
}
export default function SamplingRate(props: PropsSamplingRate) {
    const [basicData, setBasicData] = useState<Array<BoxPlotData>>(null)
    useEffect(() => {
        props.api.samplingRate(setBasicData)
    }, [])
    if (basicData == null) {
        return <CircularProgress />
    }
    console.log(basicData)
    const data = [
            {
                name: 'box',
                type: 'boxPlot',
                data: [basicData]
            }
        
    ]
    const chart_options: ApexOptions = {
        chart: {
            type: 'boxPlot',
            height: 350
        },
        colors: ['#008FFB'],
        title: {
            text: 'BoxPlot - Scatter Chart',
            align: 'left'
        },

        tooltip: {
            shared: false,
            intersect: true
        }
    }
    return (
        <div style={{padding:'0.5em'}} >
            <ReactApexChart options={chart_options} series={data} type="boxPlot"  />
        </div>

    )

}