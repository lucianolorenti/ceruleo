import React, { useEffect, useState } from 'react'
import { API, LineData } from './API';
import { CircularProgress, Grid, Paper, Typography } from "@mui/material";
import LoadableDataFrame from './DataTable';
import LinePlot from './LinePlot';
interface CategoricalFeaturesProps {
    api: API
}


export default function CategoricalFeatures(props: CategoricalFeaturesProps) {
    const [featureData, setFetureData] = useState<Array<LineData>>([])
    const [selectedNumericalFeature, setSelectedNumericalFeature] = useState<string>(null)
    const numericalFeatureSelected = (o: Object) => {
        setSelectedNumericalFeature(o['index'])
        setFetureData([])
    }

    const updateArray = (elem: LineData, i: number) => {
        elem.id = elem.id + '_' + i
        setFetureData(items => [...items, elem]);
    }
    useEffect(() => {
        if (selectedNumericalFeature != null) {
            for (let i = 0; i < 5; i++) {
                props.api.getFeatureData(selectedNumericalFeature, i, (e: LineData) => updateArray(e, i))
            }
        }
    }, [selectedNumericalFeature])
    return (
        <Grid container spacing={3}>
            <Grid item xs={12} md={12} lg={12}>
                <LoadableDataFrame selectedRowCallback={numericalFeatureSelected} title={"Numerical features"} fetcher={props.api.numericalFeatures} paginate={true} />
            </Grid>
            <Grid item xs={12} md={12} lg={12}>
                <Paper>
                {featureData.length > 0 ?

                    <LinePlot data={featureData} /> : null}
                </Paper>
            </Grid>
        </Grid>
    )
}