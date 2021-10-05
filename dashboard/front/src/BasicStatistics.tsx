import React from "react";
import { API } from "./API";
import DataTable from "./DataTable";

interface BasicStatisticsProps {
    api:API
}
export default function BasicStatistics(props: BasicStatisticsProps) {
    
    return <DataTable title="Basic statistics" fetcher={props.api.basicStatistics} />
     
}