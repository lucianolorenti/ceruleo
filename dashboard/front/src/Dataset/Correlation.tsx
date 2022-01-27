import { Box, Container, Grid, Paper } from "@mui/material";
import React, { useEffect, useState } from "react";
import { DatasetAPI as API, LineData } from "./API";
import LoadableDataFrame from "./DataTable";
import NumericalFeatureSelector from "./NumericalFeatureSelector";
import LifeSelector from "./LifeSelector";
import LinePlot from "./LinePlot";



interface CorrelationProps {
    api: API
}



export default function Correlation(props: CorrelationProps) {


    const [feature1, setFeature1] = useState<string[]>([''])
    const [feature2, setFeature2] = useState<string[]>([''])
    const [currentLife, setCurrrentLife] = useState<number>(0)

    const [feature1Data, setFeture1Data] = useState<LineData>(null)
    const [feature2Data, setFeture2Data] = useState<LineData>(null)
    useEffect(() => {
        props.api.getFeatureData(feature1[0], currentLife, setFeture1Data)
    }, [feature1, currentLife])
    useEffect(() => {
        props.api.getFeatureData(feature2[0], currentLife, setFeture2Data)
    }, [feature2, currentLife])
    let plot = null

    if ((feature1Data != null) && (feature2Data != null)) {
        plot = <LinePlot data={[feature1Data, feature2Data ]} />
    }
    return (
        <Container maxWidth={false}>


            <LoadableDataFrame title={'Correlation'} fetcher={props.api.correlation} />

            <Grid container spacing={2} style={{ marginTop: '1em' }}>
                <Grid item xs={2}>
                    <LifeSelector currentLife={currentLife} setCurrentLife={setCurrrentLife} api={props.api} />

                </Grid>

                <Grid item xs={4}>
                    <NumericalFeatureSelector features={feature1} setCurrentFeatures={setFeature1} api={props.api} />
                </Grid>
                <Grid item xs={4}>
                    <NumericalFeatureSelector features={feature2} setCurrentFeatures={setFeature2} api={props.api} />
                </Grid>
                <Grid item xs={12}>
                
                        {plot}
                </Grid>

            </Grid>

        </Container>)

}
