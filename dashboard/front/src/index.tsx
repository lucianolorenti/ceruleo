import React, { useEffect } from 'react';

import ReactDOM from 'react-dom';

import { API, APIContext } from './API'
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';



interface DashboardProps {

}
interface FeatureSelectorProps {
    setCurrentFeatures: (a: Array<string>) => void
    feature: Array<string>
    api: API
}
function FeatureSelector(props: FeatureSelectorProps) {
    const [featureList, setFeatureList] = React.useState([]);
    props.api.numerical_features(setFeatureList)
    const handleChange = (event) => {
        props.setCurrentFeatures(event.target.value);
    };

    return (<FormControl fullWidth>
        <InputLabel id="demo-simple-select-label">Age</InputLabel>
        <Select
            value={props.feature}
            label="Features to visualize"
            onChange={handleChange}
        >
            {featureList.map((elem: string, id: number) => {
                return <MenuItem value={elem}>elem</MenuItem>
            })}

        </Select>
    </FormControl>)
}
function Dashboard(props: DashboardProps) {
    const [features, setCurrentFeatures] = React.useState([]);
    return <div>
        <APIContext.Consumer> 
            {api => <FeatureSelector api={api} feature={features} setCurrentFeatures={setCurrentFeatures} />}
        </APIContext.Consumer>
    </div>;
}

ReactDOM.render(
    <APIContext.Provider value={new API("localhost", 777)} >
        <Dashboard />
    </APIContext.Provider>,
    document.getElementById('root')
);
