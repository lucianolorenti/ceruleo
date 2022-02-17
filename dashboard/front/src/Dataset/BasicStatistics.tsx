
import { Box, CircularProgress} from "@mui/material";

import React, { ReactNode, useEffect, useState } from "react";
import { DatasetAPI as API, useAPI } from "./Network/API";
import SamplingRate from './SamplingRate'

import Masonry from '@mui/lab/Masonry';
import { DataFrameInterface } from "./Network/Responses";
import FancyCard, {CardContent} from "../Components/FancyCard";

interface BasicStatisticsProps {
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
            <Masonry columns={3} spacing={5} >
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
                <SamplingRate />
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
