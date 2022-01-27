
import { Card, CircularProgress, Grid, Paper, styled, Typography } from "@mui/material";

import React, { ReactNode, useEffect, useState } from "react";
import { DatasetAPI as API, BoxPlotData } from "./API";
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
    padding: '2rem',


});
const CardTitle = styled(Typography)({
    textAlign: 'center',
    borderRadius: 'calc(.35rem - 1px) calc(.35rem - 1px) 0 0',
    backgroundColor: '#f8f9fc',
    borderBottom: '1px solid #e3e6f0',
    padding: '1rem',
    color: '#4e73df',
    fontWeight: 700,
    paddingLeft: '2rem',
    paddingRight: '2rem'

})
const FancyCard = (props: FancyCardProps) => {
    return (<Paper sx={{

        display: 'flex',
        flexDirection: 'column',
        borderRadius: '.35rem',
        boxShadow: '0 .15rem 1.75rem 0 rgba(58,59,69,.15)'
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
    const size = 4
    const style = {
        'textAlign': 'center',
        'paddingTop': '1em',
        'paddingBottom': '0.5em'
    }







    return (
        <>
            <Grid item xs>
                <FancyCard title="Number of lives">
                    <CardContent variant="h4">
                        {basicData.data[0]['Number of lives']}
                    </CardContent>
                </FancyCard>
            </Grid>
            <Grid item xs>
                <FancyCard title="Samples per life">
                    <CardContent variant="h4">
                        {basicData.data[0]['Number of samples']}
                    </CardContent>
                </FancyCard>
            </Grid>
            <Grid item xs>
                <FancyCard title='Sampling rate'>
                    <SamplingRate api={props.api} />
                </FancyCard>

            </Grid>
            <Grid item xs>
                <FancyCard title="Number of categorical fetures">
                    <CardContent variant="h4">
                        {basicData.data[0]['Number of Categorical features']}
                    </CardContent>

                </FancyCard>
            </Grid>
            <Grid item xs >
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

        <Grid container spacing={3} direction="row"
            justifyContent="flex-start"
            alignItems="flex-start">
            <BasicStatisticsCards api={props.api} />

        </Grid>
    )

}
