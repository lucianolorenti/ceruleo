import { GridCallbackDetails, GridRowParams, MuiEvent } from "@mui/x-data-grid";
import React, { useEffect, useState } from "react";
import { API, LineData } from "./API";
import LoadableDataFrame, { DataFrame } from "./DataTable";
import LinePlot from "./LinePlot";
import LoadableComponent from "./LoadableComponent";


interface BasicStatisticsProps {
    api: API
}

export default function BasicStatistics(props: BasicStatisticsProps) {
    const [featureData, setFetureData] = useState<Array<LineData>>([])
    const [selectedNumericalFeature, setSelectedNumericalFeature] = useState<string>(null)
    const numericalFeatureSelected = (o: Object) => {
        setSelectedNumericalFeature(o['index'])
        setFetureData([])
    }
    
    const updateArray = (elem:LineData, i:number) => {
        elem.id = elem.id + '_' + i
        setFetureData(items => [...items, elem]);
        
        
    }
    useEffect(() => {
        if (selectedNumericalFeature != null) {
            for (let i =0;i<5;i++) {
                props.api.getFeatureData(selectedNumericalFeature, i, (e:LineData)=> updateArray(e, i))
            }
        }
    }, [selectedNumericalFeature])
    return <div>

        <LoadableDataFrame pagination={false} show_index={false} title={"Basic statistics"} fetcher={props.api.basicStatistics} />
        <LoadableDataFrame title={"Categorical features"} fetcher={props.api.numericalFeatures} paginate={true} />
        <LoadableDataFrame selectedRowCallback={numericalFeatureSelected} title={"Numerical features"} fetcher={props.api.numericalFeatures} paginate={true} />
        
        {featureData.length > 0 ? 
        
        <LinePlot data={featureData} /> : null }
    </div>

}
