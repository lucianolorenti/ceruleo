import { AppBar, CircularProgress, Container } from '@material-ui/core';

import FormControl from '@mui/material/FormControl';
import Grid from '@mui/material/Grid';

import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import Select from '@mui/material/Select';

import React, { useEffect, useState } from 'react';

import { API, APIContext } from './API';
import FeatureDistribution from './FeatureDistribution';
import ListItemText from '@mui/material/ListItemText';


import Checkbox from '@mui/material/Checkbox';

import DataTable from './DataTable';


interface FeatureSelectorProps {
    setCurrentFeatures: (a: Array<string>) => void
    features: Array<string>
    api: API
}
function FeatureSelector(props: FeatureSelectorProps) {
    const [featureList, setFeatureList] = React.useState<Array<String>>(null);
    
    useEffect(() => {
       props.api.numericalFeatures(setFeatureList)
   
    }, [])

    const handleChange = (event) => {
        const value = event.target.value;
      
        props.setCurrentFeatures(
            typeof value === 'string' ? value.split(',') : value,
        );
    };


    return (
        <FormControl fullWidth>
            <InputLabel id="select-features">Features to visualize</InputLabel>
            <Select
                value={props.features}
                label="Features to visualize"
                multiple
                renderValue={(selected) => selected.join(', ')}
                onChange={handleChange}
            >
                {featureList?.map((elem: string, id: number) => {
                    return <MenuItem value={elem} key={id}>
                        <Checkbox checked={props.features.indexOf(elem) > -1} />
                        <ListItemText primary={elem} />
                    </MenuItem>

                })}

            </Select>
        </FormControl>
    )
}

interface DistributionAnalysisProps {
    api: API
}
export default function DistributionAnalysis(props: DistributionAnalysisProps) {
    const [features, setCurrentFeatures] = React.useState<Array<string>>([]);

    return <>
        <Grid container spacing={3}>
            <Grid item xs={12}>
                <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
                <DataTable title={'Divergence'} fetcher={props.api.KLDivergenceTable} />
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