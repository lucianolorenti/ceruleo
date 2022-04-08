import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import ListItemText from '@mui/material/ListItemText'
import MenuItem from '@mui/material/MenuItem'
import Select from '@mui/material/Select'
import React, { useEffect } from 'react'
import { DatasetAPI as API } from './Network/API'


interface LifeSelectorProps {
    setCurrentLife: (a: number) => void
    currentLife: number
    api: API
}


export default function LifeSelector(props: LifeSelectorProps) {
    const [numberOfLives, setNumberOfLives] = React.useState<number>(0);


    useEffect(() => {
        props.api.numberOfLives(setNumberOfLives)

    }, [])

    const handleChange = (event) => {
        const value = event.target.value;

        props.setCurrentLife(
            value
        );
    };


    return (
        <FormControl fullWidth>
            <InputLabel id="select-life">Selected life</InputLabel>
            <Select
                value={props.currentLife}
                label="Selected life"
                onChange={handleChange}
            >
                {[...Array(numberOfLives).keys()].map((elem: number, id: number) => {
                    return <MenuItem value={elem} key={id}>
                        <ListItemText primary={elem} />
                    </MenuItem>

                })}

            </Select>
        </FormControl>
    )
}
