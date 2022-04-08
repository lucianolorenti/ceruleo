import { Api } from "@mui/icons-material";
import React, { useState, createContext, useContext, useEffect } from "react";

import { DatasetAPI, useAPI } from "../Network/API";
import { DataFrameInterface } from "../Network/Responses";


interface FeaturesData {
    categoricals: DataFrameInterface
    numericals: DataFrameInterface
}


export const FeaturesDataContext = createContext<[FeaturesData, boolean]>(
    [{'categoricals': null, 'numericals':null}, true]
);



export const FeaturesDataProvider = (props: any) => {
    const api = useAPI()
    const [isLoading, setIsLoading] = useState(true);
    const [numericalFeatures, setNumericalFeatures] = useState<DataFrameInterface>(null);
    const [categoricalFeatures, setCategoricalsFeatures] = useState<DataFrameInterface>(null);
    useEffect(() => {
        setIsLoading(true)
        const promise1 = api.numericalFeatures().then((df: DataFrameInterface) => setNumericalFeatures(df))
        const promise2 = api.categoricalFeatures().then((df: DataFrameInterface) => setCategoricalsFeatures(df))
        Promise.all([promise1, promise2]).then(() => {
            setIsLoading(false)
        }).catch((e) => {
            console.log(e)
        })



    }, [])

    const features = {
        'numericals': numericalFeatures,
        'categoricals': categoricalFeatures
    }
    return <FeaturesDataContext.Provider value={[features, isLoading]} {...props} ></FeaturesDataContext.Provider>;



};


export function useFeatureData() {
    const context = useContext(FeaturesDataContext);
    if (context === undefined) {
        throw new Error("Context must be used within a Provider");
    }
    return context;
}


