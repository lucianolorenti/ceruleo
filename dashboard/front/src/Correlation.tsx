import { Box, Grid } from "@mui/material";
import React, { useEffect, useState } from "react";
import { API } from "./API";
import LoadableDataFrame from "./DataTable";
import FeatureSelector from "./FeatureSelector";
import LifeSelector from "./LifeSelector";
import LoadableComponent from "./LoadableComponent";


interface CorrelationProps {
    api: API
}

export default function BasicStatistics(props: CorrelationProps) {


    const [feature1, setFeature1] = useState<string[]>([''])
    const [feature2, setFeature2] = useState<string[]>([''])
    const [currentLife, setCurrrentLife] = useState<number>(0)

    const [feature1Data, setFeture1Data] = useState<number[]>([])
    const [feature2Data, setFeture2Data] = useState<number[]>([])
    useEffect(() => {
        props.api.getFeatureData(feature1[0], currentLife, setFeture1Data)
    }, [feature1, currentLife])
    useEffect(() => {
        props.api.getFeatureData(feature2[0], currentLife, setFeture2Data)
    }, [feature2, currentLife])
    return <>
        <Grid container spacing={2}>
            <Grid item xs={2}>
                <LoadableDataFrame title={'Correlation'} fetcher={props.api.correlation} />
            </Grid>
        </Grid>

        <Grid container spacing={2}>
            <Grid item xs={2}>
                <LifeSelector currentLife={currentLife} setCurrentLife={setCurrrentLife} api={props.api} />

            </Grid>

            <Grid item xs={4}>
                <FeatureSelector features={feature1} multiple setCurrentFeatures={setFeature1} api={props.api} />
            </Grid>
            <Grid item xs={4}>
                <FeatureSelector features={feature2} multiple setCurrentFeatures={setFeature2} api={props.api} />
            </Grid>

        </Grid>

    </>

}
