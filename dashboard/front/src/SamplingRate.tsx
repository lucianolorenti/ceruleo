
import { Checkbox, CircularProgress, FormControl, FormControlLabel, FormGroup, Grid, InputLabel, MenuItem, Select, Typography } from "@mui/material";

import React, { useEffect, useState } from "react";
import { API, BoxPlot, BoxPlotData } from "./API";
import { VictoryChart, VictoryBoxPlot, VictoryTheme, VictoryTooltip, VictoryScatter, VictoryZoomContainer, VictoryAxis } from 'victory';

interface PropsSamplingRate {
    api: API
}
export default function SamplingRate(props: PropsSamplingRate) {
    const [unit, setUnit] = React.useState('s');

    const handleUnitChange = (event) => {
        setUnit(event.target.value);
    };

    const [basicData, setBasicData] = useState<BoxPlot>(null)
    useEffect(() => {
        props.api.samplingRate(setBasicData, unit)
    }, [])
    useEffect(() => {
        props.api.samplingRate(setBasicData, unit)
    }, [unit])
    if (basicData == null) {
        return <CircularProgress />
    }
    const max = basicData.boxplot[0].max
    const min = basicData.boxplot[0].min


    return (
        <Grid container spacing={0}>
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
                <div >
                    <VictoryChart padding={50} height={300} containerComponent={<VictoryZoomContainer zoomDomain={{ y: [min, max + (max - min) * 0.5] }} zoomDimension="y" allowPan={true} />} theme={VictoryTheme.material}       >
                        <VictoryAxis

                            label={unit}
                            dependentAxis
                        />
                        <VictoryBoxPlot
                            animate={{
                                duration: 2000,
                                onLoad: { duration: 1000 }
                            }}
                            boxWidth={40}
                            data={basicData.boxplot}
                        />
                        <VictoryScatter
                            style={{ data: { fill: "#c43a31" } }}
                            categories={{ x: ["Sampling rate"] }}
                            size={1}
                            data={basicData.outliers[0]}
                        />
                    </VictoryChart>
                </div>
            </Grid>
        </Grid>

    )

}