import React from "react";
import { API } from "./API";
import LoadableDataFrame, { DataFrame } from "./DataTable";
import LoadableComponent from "./LoadableComponent";


interface BasicStatisticsProps {
    api:API
}

export default function BasicStatistics(props: BasicStatisticsProps) {

    return <div>
        
        <LoadableDataFrame title={"Basic statistics"} fetcher={props.api.basicStatistics} />
        <LoadableDataFrame title={"Categorical features"} fetcher={props.api.numericalFeatures} paginate={true} />
        <LoadableDataFrame title={"Numerical features"} fetcher={props.api.numericalFeatures} paginate={true} />
        <div>
            Sampling rate
            
        </div>
    </div> 
     
}
