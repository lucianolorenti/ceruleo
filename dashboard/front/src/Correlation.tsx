import { Box, Container, Grid, Paper } from "@mui/material";
import React, { useEffect, useState } from "react";
import { API, LineData } from "./API";
import LoadableDataFrame from "./DataTable";
import FeatureSelector from "./FeatureSelector";
import LifeSelector from "./LifeSelector";

import { LineSvgProps, ResponsiveLine } from '@nivo/line'

interface CorrelationProps {
    api: API
}



export default function Correlation(props: CorrelationProps) {


    const [feature1, setFeature1] = useState<string[]>([''])
    const [feature2, setFeature2] = useState<string[]>([''])
    const [currentLife, setCurrrentLife] = useState<number>(0)

    const [feature1Data, setFeture1Data] = useState<LineData>(null)
    const [feature2Data, setFeture2Data] = useState<LineData>(null)
    useEffect(() => {
        props.api.getFeatureData(feature1[0], currentLife, setFeture1Data)
    }, [feature1, currentLife])
    useEffect(() => {
        props.api.getFeatureData(feature2[0], currentLife, setFeture2Data)
    }, [feature2, currentLife])
    let plot = null
    console.log(feature1Data)
    console.log(feature2Data)
    if ((feature1Data != null) && (feature2Data != null)) {
        plot = <ResponsiveLine

            data={[feature1Data, feature2Data]}
            margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
            xScale={{ type: 'linear', max: 'auto', min: 'auto' }}
            yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false }}
            yFormat=" >-.2f"
            curve='monotoneX'
            enablePoints={false}
            axisTop={null}
            axisRight={null}
            axisBottom={{

                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'transportation',
                legendOffset: 36,
                legendPosition: 'middle'
            }}
            axisLeft={{

                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'count',
                legendOffset: -40,
                legendPosition: 'middle'
            }}
            pointSize={10}
            pointColor={{ theme: 'background' }}
            pointBorderWidth={2}
            pointBorderColor={{ from: 'serieColor' }}
            pointLabelYOffset={-12}
            useMesh={true}
            legends={[
                {
                    anchor: 'bottom-right',
                    direction: 'column',
                    justify: false,
                    translateX: 100,
                    translateY: 0,
                    itemsSpacing: 0,
                    itemDirection: 'left-to-right',
                    itemWidth: 80,
                    itemHeight: 20,
                    itemOpacity: 0.75,
                    symbolSize: 12,
                    symbolShape: 'circle',
                    symbolBorderColor: 'rgba(0, 0, 0, .5)',
                    effects: [
                        {
                            on: 'hover',
                            style: {
                                itemBackground: 'rgba(0, 0, 0, .03)',
                                itemOpacity: 1
                            }
                        }
                    ]
                }
            ]}

        />
    }
    console.log(plot)
    return (
        <Container maxWidth={false}>


            <LoadableDataFrame title={'Correlation'} fetcher={props.api.correlation} />

            <Grid container spacing={2} style={{ marginTop: '1em' }}>
                <Grid item xs={2}>
                    <LifeSelector currentLife={currentLife} setCurrentLife={setCurrrentLife} api={props.api} />

                </Grid>

                <Grid item xs={4}>
                    <FeatureSelector features={feature1} setCurrentFeatures={setFeature1} api={props.api} />
                </Grid>
                <Grid item xs={4}>
                    <FeatureSelector features={feature2} setCurrentFeatures={setFeature2} api={props.api} />
                </Grid>
                <Grid item xs={12}>
                    <div style={{ height: '400px' }} >
                        {plot}
                    </div>
                </Grid>

            </Grid>

        </Container>)

}
