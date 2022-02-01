import { Api } from "@mui/icons-material";
import React, { useState, createContext, useContext, useEffect } from "react";
import { DatasetAPI, useAPI } from "../Network/API";


interface FeatureNames {
    categoricals: Array<string>
    numericals: Array<string>
}


export const FeaturesContext = createContext<[FeatureNames, boolean]>(
    [{'categoricals': [], 'numericals':[]}, true]
);



export const FeaturesProvider = (props: any) => {
    const api = useAPI()
    const [isLoading, setIsLoading] = useState(true);
    const [numericalFeatures, setNumericalFeatures] = useState([]);
    const [categoricalFeatures, setCategoricalsFeatures] = useState([]);
    useEffect(() => {
        setIsLoading(true)
        const promise1 = api.numericalFeaturesList().then((list: Array<string>) => setNumericalFeatures(list))
        const promise2 = api.categoricalFeaturesList().then((list: Array<string>) => setCategoricalsFeatures(list))
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
    return <FeaturesContext.Provider value={[features, isLoading]} {...props} ></FeaturesContext.Provider>;



};


export function useFeatureNames() {
    const context = useContext(FeaturesContext);
    if (context === undefined) {
        throw new Error("Context must be used within a Provider");
    }
    return context;
}


