

import Grid from '@mui/material/Grid';


import Paper from '@mui/material/Paper';

import React, { useEffect, useState } from 'react';

import { API, APIContext } from './API';
import FeatureDistribution from './FeatureDistribution';

import LoadableComponent from './LoadableComponent';
import { DataFrame } from './DataTable';
import NumericalFeatureSelector from './NumericalFeatureSelector'


interface DistributionAnalysisProps {
    api: API
}
//const StatisticsTable = LoadableComponent("Divergence", DataFrame)
//props.api.KLDivergenceTable,
export default function DistributionAnalysis(props: DistributionAnalysisProps) {
    const [features, setCurrentFeatures] = React.useState<Array<string>>([]);
   
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
                  <NumericalFeatureSelector api={props.api} multiple={true} features={features} setCurrentFeatures={setCurrentFeatures} />
                    <FeatureDistribution api={props.api} features={features} />
                  
                </Paper>
            </Grid>


        </Grid>

    </>
}
//   
//  