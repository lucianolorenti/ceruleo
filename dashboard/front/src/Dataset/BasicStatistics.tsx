
import { Box, Card, CircularProgress, Grid, Paper, styled, Typography } from "@mui/material";

import React, { ReactNode, useEffect, useState } from "react";
import { DatasetAPI as API, useAPI } from "./Network/API";
import SamplingRate from './SamplingRate'

import Masonry from '@mui/lab/Masonry';
import { DataFrameInterface } from "./Network/Responses";

interface BasicStatisticsProps {
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
        borderRadius: '.35rem',
        boxShadow: '0 .15rem 1.75rem 0 rgba(58,59,69,.15)'
    }}>
        <CardTitle variant="h5" > {props.title} </CardTitle>
        {props.children}

    </Paper>
    )
}




export default function BasicStatistics(props: BasicStatisticsProps) {
    const api = useAPI()
    const [basicData, setBasicData] = useState<DataFrameInterface>(null)
    useEffect(() => {
        api.basicStatistics(setBasicData)
    }, [])
    if (basicData == null) {
        return <CircularProgress />
    }
    return (
        <Box sx={{ width: '80vw', minHeight: 377 }}>
            <Masonry columns={3} spacing={1} >
            <FancyCard title="Number of lives">
                <CardContent variant="h4">
                    {basicData.data[0]['Number of lives']}
                </CardContent>
            </FancyCard>

            <FancyCard title="Samples per life">
                <CardContent variant="h4">
                    {basicData.data[0]['Number of samples']}
                </CardContent>
            </FancyCard>

            <FancyCard title='Sampling rate' >
                <SamplingRate api={api} />
            </FancyCard>


            <FancyCard title="Number of categorical fetures">
                <CardContent variant="h4">
                    {basicData.data[0]['Number of Categorical features']}
                </CardContent>

            </FancyCard>

            <FancyCard title="Numerical features">
                <CardContent variant="h4">
                    {basicData.data[0]['Number of Numerical features']}
                </CardContent>
            </FancyCard>



            </Masonry>
        </Box>
    )

}
