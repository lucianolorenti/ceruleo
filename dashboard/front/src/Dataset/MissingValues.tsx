import { Box, Container, Grid, Paper } from "@mui/material";
import React, { useEffect, useState } from "react";
import { DatasetAPI as API, useAPI } from "./Network/API";
import LoadableDataFrame from "../Components/DataTable";
import NumericalFeatureSelector from "./NumericalFeatureSelector";
import LifeSelector from "./LifeSelector";
import LinePlot from "../Graphics/LinePlot";
import { LineData } from "./Network/Responses";
import { PlotData } from "../Graphics/Types";



interface MissingValuesProps {
    
}



export default function MissingValues(props: MissingValuesProps) {
    const api  = useAPI()

   
    return (
        <Container maxWidth={false}>





        </Container>)

}

//             <LoadableDataFrame title={'Correlation'} fetcher={props.api.correlation} />