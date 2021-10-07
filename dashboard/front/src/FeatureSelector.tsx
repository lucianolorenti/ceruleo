
import Checkbox from '@mui/material/Checkbox'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import ListItemText from '@mui/material/ListItemText'
import MenuItem from '@mui/material/MenuItem'
import Select from '@mui/material/Select'
import React, { useEffect } from 'react'
import { API } from './API'


interface FeatureSelectorProps {
    setCurrentFeatures: (a: Array<string>) => void
    features: Array<string>
    api: API
    multiple: boolean
}


export default function FeatureSelector(props: FeatureSelectorProps) {
    const [featureList, setFeatureList] = React.useState<Array<String>>(null);
    
    useEffect(() => {
       props.api.numericalFeaturesDistribution(setFeatureList)
   
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
                multiple={props.multiple}
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