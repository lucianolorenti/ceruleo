import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { DatasetAPI as API, useAPI } from './Network/API';
import { Box, CircularProgress, Grid, Paper, styled, Typography } from "@mui/material";
import LoadableDataFrame from '../Components/DataTable';

import ReactApexChart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';
import AutoSizer from 'react-virtualized-auto-sizer';
import { LineData, PlotData, SeriesData } from './Network/Responses';
import DistributionPlot from '../Graphics/DistributionPlot';

import { useFeatureNames } from './Store/FeatureNames';
import { useFeatureData } from './Store/FeatureTables';
import LinePlot from '../Graphics/LinePlot';


interface NumericalFeaturesProps {

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

            api.getFeatureData(selectedNumericalFeature, i).then((e: PlotData) => updateArray(e, i))


        }

    }, [selectedNumericalFeature])


    const ppa = (elem) => {

    }

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

                            {
                                (selectedNumericalFeature == null) ?
                                    (<Typography variant="h4" component="div" gutterBottom>
                                        Please select a feature to start
                                    </Typography>)
                                    :
                                    (featurePlotData.length > 0) ?
                                        <LinePlot
                                            feature={selectedNumericalFeature}
                                            series={featurePlotData}                                            
                                            height={350} />
                                        :
                                        <CircularProgress />

                            }

                        </Paper>
                    </Grid>


                    <Grid item xs={12} md={12} lg={12}>
                        <Paper  >

                            {
                                (selectedNumericalFeature == null) ?
                                    (<Typography variant="h4" component="div" gutterBottom>
                                        Please select a feature to start
                                    </Typography>)
                                    :
                                    (featurePlotData.length > 0) ?
                                        <DistributionPlot
                                            fetch_data={api.getDistributionData}
                                            selectedFeature={selectedNumericalFeature}
                                            title={selectedNumericalFeature}
                                        />
                                        :
                                        <CircularProgress />

                            }




                        </Paper>
                    </Grid>
                </Grid>
            </Grid>
        </Grid>

    )
}

