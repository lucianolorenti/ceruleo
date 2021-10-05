import React, { useState } from "react";
import { API } from "./API";
import { DataFrame } from "./DataTable";
import FeatureSelector from "./FeatureSelector";
import LoadableComponent from "./LoadableComponent";


interface CorrelationProps {
    api:API
}

export default function BasicStatistics(props: CorrelationProps) {
    const CorrelationTable = LoadableComponent("Correlation", props.api.correlation, DataFrame)
    
    const [feature1, setFeature1] = useState<string[]>([''])
    const [feature2, setFeature2] = useState<string[]>([''])
    return <div>
        <CorrelationTable />
        <div>
            <div>
                <FeatureSelector features={feature1} setCurrentFeatures={setFeature1} api={props.api} />
            </div>
            <div>
                <FeatureSelector  features={feature2} setCurrentFeatures={setFeature2} api={props.api}/>
            </div>
        </div>
    </div> 
     
}
