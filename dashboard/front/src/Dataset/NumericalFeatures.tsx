import React, { useEffect, useState } from 'react'
import { DatasetAPI as API, LineData } from './API';
import { CircularProgress, Grid, Paper, Typography } from "@mui/material";
import LoadableDataFrame from './DataTable';
import LinePlot from './LinePlot';
import ReactApexChart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';
interface NumericalFeaturesProps {
    api: API
}
function build_options(title: string): ApexOptions {
    return {
        chart: {
            height: 350,
            type: "line",
            stacked: false
        },
        dataLabels: {
            enabled: false
        },

        title: {
            text: title,
            align: 'left',
            offsetX: 110
        },

        yaxis: [
            {
                axisTicks: {
                    show: true,
                },
                axisBorder: {
                    show: true,
                    color: '#008FFB'
                },
                labels: {
                    style: {
                        colors: '#008FFB',
                    }
                },
                tooltip: {
                    enabled: true
                }
            }

        ],
        tooltip: {
            fixed: {
                enabled: true,
                position: 'topLeft', // topRight, topLeft, bottomRight, bottomLeft
                offsetY: 30,
                offsetX: 60
            },
        },
        legend: {
            horizontalAlign: 'left',
            offsetX: 40
        }
    }
}

export default function NumericalFeatures(props: NumericalFeaturesProps) {
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

    const series = featureData.map((elem: LineData, index: number) => {
        return {
            name: selectedNumericalFeature + ' ' + index.toString(),
            type: 'line',
            data: elem.data

        }
    })

    return (
        <Grid container spacing={3}>
            <Grid item xs={12} md={12} lg={12}>
                <LoadableDataFrame selectedRowCallback={numericalFeatureSelected} title={"Numerical features"} fetcher={props.api.numericalFeatures} paginate={true} />
            </Grid>
            <Grid item xs={12} md={12} lg={12}>
                <Paper>
                    {featureData.length > 0 ?

                        <ReactApexChart options={build_options(selectedNumericalFeature)} series={series} type="line" height={350} /> : null}
                </Paper>
            </Grid>
        </Grid>
    )
}