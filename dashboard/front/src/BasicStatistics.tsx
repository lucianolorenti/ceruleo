
import { Card, CircularProgress, Grid, Paper, styled, Typography } from "@mui/material";

import React, { ReactNode, useEffect, useState } from "react";
import { API, BoxPlotData } from "./API";
import LoadableDataFrame, { DataFrame, DataFrameInterface } from "./DataTable";
import SamplingRate from './SamplingRate'

interface BasicStatisticsProps {
    api: API
}

interface BasicStatisticsCardsProps {
    api: API
}

interface FancyCardProps {
    title: string
    children: ReactNode;

}
const CardContent = styled(Typography)({
    textAlign: 'center',
    paddingTop: '1em',
    paddingBottom: '0.5em',
});
const CardTitle = styled(Typography)({
    textAlign: 'center',
    backgroundColor: '#DADAFF',
    marginTop: '-0.7em',
    marginLeft: '-0.7em',
    marginRight: '-0.7em',
    marginBottom: '0.3em',
    padding: '0.5em'
})
const FancyCard = (props: FancyCardProps) => {
    return (<Paper sx={{
        p: 2,
        display: 'flex',
        flexDirection: 'column'
    }}>
        <CardTitle variant="h5" > {props.title} </CardTitle>
        {props.children}

    </Paper>
    )
}





const BasicStatisticsCards = (props: BasicStatisticsCardsProps) => {
    const [basicData, setBasicData] = useState<DataFrameInterface>(null)
    useEffect(() => {
        props.api.basicStatistics(setBasicData)
    }, [])
    if (basicData == null) {
        return <CircularProgress />
    }
    const size = 3
    const style = {
        'textAlign': 'center',
        'paddingTop': '1em',
        'paddingBottom': '0.5em'
    }







    return (
        <>
            <Grid item xs={size} md={size} lg={size}>
                <FancyCard title="Number of lives">
                    <CardContent variant="h4">
                        {basicData.data[0]['Number of lives']}
                    </CardContent>
                </FancyCard>
            </Grid>
            <Grid item xs={size} md={size} lg={size}>
                <FancyCard title="Samples per life">
                    <CardContent variant="h4">
                        {basicData.data[0]['Number of samples']}
                    </CardContent>
                </FancyCard>
            </Grid>
            <Grid item xs={size} md={size} lg={size}>
                <FancyCard title="Number of categorical fetures">
                    <CardContent variant="h4">
                        {basicData.data[0]['Number of Categorical features']}
                    </CardContent>

                </FancyCard>
            </Grid>
            <Grid item xs={size} md={size} lg={size}>
                <FancyCard title="Numerical features">
                    <CardContent variant="h4">
                        {basicData.data[0]['Number of Numerical features']}
                    </CardContent>
                </FancyCard>
            </Grid>

        </>)

}

export default function BasicStatistics(props: BasicStatisticsProps) {

    return (

        <Grid container spacing={3}>
            <BasicStatisticsCards api={props.api} />
            <Grid item xs={4}>
                <FancyCard title='Sampling rate'>
                    <SamplingRate api={props.api} />
                </FancyCard>

            </Grid>
        </Grid>
    )

}
