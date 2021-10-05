import React from "react";
import { API } from "./API";
import { DataFrame } from "./DataTable";
import LoadableComponent from "./LoadableComponent";


interface CorrelationProps {
    api:API
}

export default function BasicStatistics(props: CorrelationProps) {
    const CorrelationTable = LoadableComponent("Correlation", props.api.correlation, DataFrame)
    
    return <div>
        <CorrelationTable />
    </div> 
     
}
