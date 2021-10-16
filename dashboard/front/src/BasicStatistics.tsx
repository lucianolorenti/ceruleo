
import { CircularProgress, Grid, Paper, styled, Typography } from "@mui/material";
import { GridCallbackDetails, GridRowParams, MuiEvent } from "@mui/x-data-grid";
import { alignBox } from "@nivo/core";
import React, { ReactNode, useEffect, useState } from "react";
import { API, LineData } from "./API";
import LoadableDataFrame, { DataFrame, DataFrameInterface } from "./DataTable";
import LinePlot from "./LinePlot";
import LoadableComponent from "./LoadableComponent";

import ReactApexChart from "react-apexcharts";
import { ApexOptions } from "apexcharts";


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
    margin: '-0.7em',
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

    const data = {

        series: [
            {
                name: 'box',
                type: 'boxPlot',
                data: [
                    {
                        x: new Date('2017-01-01').getTime(),
                        y: [54, 66, 69, 75, 88]
                    },
                    {
                        x: new Date('2018-01-01').getTime(),
                        y: [43, 65, 69, 76, 81]
                    },
                    {
                        x: new Date('2019-01-01').getTime(),
                        y: [31, 39, 45, 51, 59]
                    },
                    {
                        x: new Date('2020-01-01').getTime(),
                        y: [39, 46, 55, 65, 71]
                    },
                    {
                        x: new Date('2021-01-01').getTime(),
                        y: [29, 31, 35, 39, 44]
                    }
                ]
            },
            {
                name: 'outliers',
                type: 'scatter',
                data: [
                    {
                        x: new Date('2017-01-01').getTime(),
                        y: 32
                    },
                    {
                        x: new Date('2018-01-01').getTime(),
                        y: 25
                    },
                    {
                        x: new Date('2019-01-01').getTime(),
                        y: 64
                    },
                    {
                        x: new Date('2020-01-01').getTime(),
                        y: 27
                    },
                    {
                        x: new Date('2020-01-01').getTime(),
                        y: 78
                    },
                    {
                        x: new Date('2021-01-01').getTime(),
                        y: 15
                    }
                ]
            }
        ]
    }
    const chart_options :ApexOptions = {
            chart: {
                type: 'boxPlot',
                height: 350
            },
            colors: ['#008FFB', '#FEB019'],
            title: {
                text: 'BoxPlot - Scatter Chart',
                align: 'left'
            },

            tooltip: {
                shared: false,
                intersect: true
            }
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
        <Grid item xs={12}>
            Sampling rate
            <div id="chart">
                <ReactApexChart options={chart_options} series={data.series} type="boxPlot" height={350} />
            </div>

        </Grid>
    </>)

}

export default function BasicStatistics(props: BasicStatisticsProps) {

    return (

        <Grid container spacing={3}>
            <BasicStatisticsCards api={props.api} />
        </Grid>
    )

}
