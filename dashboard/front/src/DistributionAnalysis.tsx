import { AppBar, CircularProgress, Container } from '@material-ui/core';



import Grid from '@mui/material/Grid';


import Paper from '@mui/material/Paper';

import React, { useEffect, useState } from 'react';

import { API, APIContext } from './API';
import FeatureDistribution from './FeatureDistribution';

import LoadableComponent from './LoadableComponent';
import { DataFrame } from './DataTable';
import FeatureSelector from './FeatureSelector'


interface DistributionAnalysisProps {
    api: API
}
export default function DistributionAnalysis(props: DistributionAnalysisProps) {
    const [features, setCurrentFeatures] = React.useState<Array<string>>([]);
    const StatisticsTable = LoadableComponent("Divergence", props.api.KLDivergenceTable, DataFrame)

    return <>
        <Grid container spacing={3}>
            <Grid item xs={12}>
                <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
                
                </Paper>
            </Grid>
            <Grid item xs={12} md={12} lg={12}>
                <Paper
                    sx={{
                        p: 2,
                        display: 'flex',
                        flexDirection: 'column',
                        height: 600,
                    }}
                >
                  <FeatureSelector api={props.api} features={features} setCurrentFeatures={setCurrentFeatures} />
                    <FeatureDistribution api={props.api} features={features} />
                  
                </Paper>
            </Grid>


        </Grid>

    </>
}
//   
//  