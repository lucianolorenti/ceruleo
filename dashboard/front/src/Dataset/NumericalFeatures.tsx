import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { DatasetAPI as API, useAPI } from './Network/API';
import { Box, CircularProgress, Grid, Paper, Typography } from "@mui/material";
import LoadableDataFrame from '../Components/DataTable';

import ReactApexChart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';
import AutoSizer from 'react-virtualized-auto-sizer';
import { LineData, SeriesData } from './Network/Responses';
import DistributionPlot from '../Graphics/DistributionPlot';
import { PlotData } from '../Graphics/Types';
import { useFeatureNames } from './Store/FeatureNames';
import { useFeatureData } from './Store/FeatureTables';


interface NumericalFeaturesProps {
   
}
function build_options(title: string): ApexOptions {
    return {
        chart: {
            height: 350,
            type: "line",
            stacked: false,
            animations: {
                enabled: false
            }
        },
        dataLabels: {
            enabled: false
        },

        title: {
            text: title,
            align: 'left',
            offsetX: 110
        },
        markers: {
            size: 0
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
        xaxis: {
            tickAmount: 20
        },
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
    const api = useAPI()
    const [featureData, featureDataLoading] = useFeatureData()
    const [featurePlotData, setFeaturePlotData] = useState<Array<PlotData>>([])

    const [selectedNumericalFeature, setSelectedNumericalFeature] = useState<string>(null)
    const numericalFeatureSelected = (o: Object) => {
        setSelectedNumericalFeature(o['index'])
        setFeaturePlotData([])

    }


    const updateArray = (elem: PlotData, i: number) => {
        setFeaturePlotData(items => [...items, elem]);
    }


    useEffect(() => {
        if (selectedNumericalFeature == null) {
            return
        }
        for (let i = 0; i < 10; i++) {

            api.getFeatureData(selectedNumericalFeature, i, (e: PlotData) => updateArray(e, i))


        }

    }, [selectedNumericalFeature])




    return (

        <Grid container spacing={3} >
            <Grid item xs={4} md={4} lg={4} style={{ 'height': '90vh' }}>
                <AutoSizer>
                    {({ height, width }) => {
                        console.log(height)
                        const pageSize = Math.floor((height - 192) / 30);

                        return (
                            <div style={{ height: `${height}px`, width: `${width}px`, overflowY: 'auto' }}>
                                <LoadableDataFrame
                                    loading={featureDataLoading}
                                    selectedRowCallback={numericalFeatureSelected}
                                    title={"Numerical features"}
                                    data={featureData.numericals}
                                    paginate={true}
                                    pageSize={pageSize} />
                            </div>
                        )
                    }
                    }
                </AutoSizer>

            </Grid>
            <Grid item xs={8} md={8} lg={8}>
                <Grid container spacing={3} >
                    <Grid item xs={12} md={12} lg={12}>
                        <Paper >
                            {featurePlotData.length > 0 ?
                                <ReactApexChart
                                    options={build_options(selectedNumericalFeature)}
                                    series={featurePlotData}
                                    type="line"
                                    height={350} /> : <CircularProgress />}
                        </Paper>
                    </Grid>


                    <Grid item xs={12} md={12} lg={12}>
                        <Paper >
                            <DistributionPlot
                                fetch_data={api.getDistributionData}
                                selectedFeature={selectedNumericalFeature}
                                title={selectedNumericalFeature}
                            />


                        </Paper>
                    </Grid>
                </Grid>
            </Grid>
        </Grid>

    )
}

