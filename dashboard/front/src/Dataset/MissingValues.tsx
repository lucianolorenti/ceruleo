import { Box, CircularProgress, Container, Grid, Paper, Typography } from "@mui/material";
import React, { ReactNode, useEffect, useState } from "react";
import { DatasetAPI as API, useAPI } from "./Network/API";
import LoadableDataFrame from "../Components/DataTable";

import { DataFrameInterface, LineData, PlotData } from "./Network/Responses";
import { GridRenderCellParams } from "@mui/x-data-grid";
import LinearProgress from '@mui/material/LinearProgress';
import ReactApexChart from 'react-apexcharts';
import { ApexOptions } from "apexcharts";
import ScoreBar from "../Components/ScoreBar/ScoreBar";
import LinePlot from "../Graphics/LinePlot";


interface MissingValuesProps {

}





export default function MissingValues(props: MissingValuesProps) {
    const api = useAPI()
    const [loading, setLoading] = useState(true)
    const [missingValuesData, setMissingValuesData] = useState<DataFrameInterface>(null)
    const [selectedNumericalFeature, setSelectedNumericalFeature] = useState<string>(null)
    const [featurePlotData, setFeaturePlotData] = useState<Array<PlotData>>([])
    useEffect(() => {
        api.getMissingValues().then((missingValuesData: DataFrameInterface) => {
            setMissingValuesData(missingValuesData)
        }).then(() => {
            setLoading(false)
        })

    }, [])
    const updateArray = (elem: PlotData, i: number) => {
        setFeaturePlotData(items => [...items, elem]);
    }
    const numericalFeatureSelected = (o: Object) => {
        console.log(o['Feature'])
        setSelectedNumericalFeature(o['Feature'])
        setFeaturePlotData([])

    }
    useEffect(() => {
        if (selectedNumericalFeature == null) {
            return
        }
        for (let i = 0; i < 10; i++) {

            api.getFeatureData(selectedNumericalFeature, i).then((e: PlotData) => updateArray(e, i))


        }

    }, [selectedNumericalFeature])

    if (loading) {
        return <CircularProgress />
    }

    return (
        <Container maxWidth={false}>
            <Grid container>
                <Grid item xs={12}>
                    <Paper style={{ 'padding': '2em', 'margin': '1em' }}>
                        <LoadableDataFrame
                            loading={loading}
                            title={"Null proportion"}
                            data={missingValuesData}
                            selectedRowCallback={numericalFeatureSelected}
                            paginate={true}
                            show_index={false}
                            pageSize={8}
                            renderCells={
                                {

                                    'Null proportion': (params: GridRenderCellParams): ReactNode => {
                                        const values = String(params.value.valueOf()).split(',').map((elem: string) => +elem * 100)

                                        return <ScoreBar min={+values[0]} max={+values[1]} mean={+values[2]} />

                                    }
                                    ,

                                    'Entropy': (params: GridRenderCellParams): ReactNode => {
                                        const values = String(params.value.valueOf()).split(',').map((elem: string) => +elem * 100)

                                        return <ScoreBar min={+values[0]} max={+values[1]} mean={+values[2]} />

                                    }
                                }
                            } />
                    </Paper>

                </Grid>

                <Grid item xs={12}>

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




            </Grid>





        </Container>)

}

//             <LoadableDataFrame title={'Correlation'} fetcher={props.api.correlation} />