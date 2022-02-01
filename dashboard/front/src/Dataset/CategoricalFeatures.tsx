import React, { useEffect, useState } from 'react'
import { DatasetAPI as API } from './Network/API';
import { CircularProgress, Grid, Paper, Typography } from "@mui/material";
import LoadableDataFrame from '../Components/DataTable';
import LinePlot from '../Graphics/LinePlot';
import { LineData } from './Network/Responses';
import { PlotData } from '../Graphics/Types';
interface CategoricalFeaturesProps {
    api: API
}


export default function CategoricalFeatures(props: CategoricalFeaturesProps) {
    const [featureData, setFetureData] = useState<Array<PlotData>>([])
    const [selectedNumericalFeature, setSelectedNumericalFeature] = useState<string>(null)
    const numericalFeatureSelected = (o: Object) => {
        setSelectedNumericalFeature(o['index'])
        setFetureData([])
    }

    const updateArray = (elem: PlotData, i: number) => {
       
        setFetureData(items => [...items, elem]);
    }
    useEffect(() => {
        if (selectedNumericalFeature != null) {
            for (let i = 0; i < 5; i++) {
                props.api.getFeatureData(selectedNumericalFeature, i, (e: PlotData) => updateArray(e, i))
            }
        }
    }, [selectedNumericalFeature])
    return (
        <Grid container spacing={3}>
            <Grid item xs={12} md={12} lg={12}>
                
            </Grid>
            <Grid item xs={12} md={12} lg={12}>
                <Paper>
           
                </Paper>
            </Grid>
        </Grid>
    )
}
//<LoadableDataFrame selectedRowCallback={numericalFeatureSelected} title={"Numerical features"} fetcher={props.api.numericalFeatures} paginate={true} />