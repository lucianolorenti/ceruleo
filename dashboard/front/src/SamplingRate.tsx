
import { Checkbox, CircularProgress, FormControl, FormControlLabel, FormGroup, Grid, InputLabel, MenuItem, Select, Typography } from "@mui/material";

import React, {  useEffect, useState } from "react";
import { API, BoxPlotData } from "./API";


import ReactApexChart from "react-apexcharts";
import { ApexOptions } from "apexcharts";

interface PropsSamplingRate {
    api: API
}
export default function SamplingRate(props: PropsSamplingRate) {
    const [unit, setUnit] = React.useState('');

    const handleUnitChange = (event) => {
        setUnit(event.target.value);
    };
  
    const [basicData, setBasicData] = useState<Array<BoxPlotData>>(null)
    useEffect(() => {
        props.api.samplingRate(setBasicData)
    }, [])
    if (basicData == null) {
        return <CircularProgress />
    }
    const data = [
        {
            name: 'box',
            type: 'boxPlot',
            data: basicData[0].data
        },
        {
            name: 'outliers',
            type: 'scatter',
            data: basicData[0].outliers
        }

    ]
    const chart_options: ApexOptions = {
        chart: {
            type: 'scatter',
            height: 450,
           
        },
       

        colors: ['#008FFB', '#FEB019'],
        xaxis: {
            type: 'categories',
            categories: ['Sample rate'],
            sorted: false,
            overwriteCategories: ['Sample rate']
         
           
        },
        tooltip: {
            shared: false,
            intersect: true
        }
    }
    return (
        <Grid container spacing={3}>
            <Grid item sm={6}>
                <FormControl fullWidth>
                    <InputLabel id="sample-rate-units">Unit</InputLabel>
                    <Select
                        labelId="sample-rate-unit"
                        value={unit}
                        label="Age"
                        onChange={handleUnitChange}
                    >
                        <MenuItem value={''}>Not specified</MenuItem>
                        <MenuItem value={'s'}>Seconds</MenuItem>
                        <MenuItem value={'m'}>Minutes</MenuItem>
                    </Select>
                </FormControl>
            </Grid>
            <Grid item sm={12}>
                <Typography>
                    {basicData[0].data.y[3]} [{unit}]
                </Typography>
            </Grid>
            <Grid item sm={12}>
                <div >
                    <ReactApexChart options={chart_options} series={data} type="boxPlot" />
                </div>
            </Grid>
            </Grid>

    )

}