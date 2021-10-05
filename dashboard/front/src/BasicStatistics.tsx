import React from "react";
import { API } from "./API";
import { DataFrame } from "./DataTable";
import LoadableComponent from "./LoadableComponent";


interface BasicStatisticsProps {
    api:API
}

export default function BasicStatistics(props: BasicStatisticsProps) {
    const StatisticsTable = LoadableComponent("Basic statistics", props.api.basicStatistics, DataFrame)
    const NumericalFeaturesTable = LoadableComponent("Numerical Features", props.api.numericalFeatures, DataFrame)
    return <div>
        <StatisticsTable />
        <NumericalFeaturesTable paginate={true} />
    </div> 
     
}
