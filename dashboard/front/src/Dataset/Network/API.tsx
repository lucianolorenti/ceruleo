import React, { useContext } from "react";
import urlcat from "urlcat";

import { Stats } from '@visx/mock-data/lib/generators/genStats';
import { BoxPlot, DataFrameInterface, LineData, PlotData, SeriesData } from "./Responses";




export class DatasetAPI {
  url: string;
  port: Number;
  endpoint: string;

  constructor(url: string, port: Number) {
    this.url = url;
    this.port = port;
    this.endpoint = url + ":" + port + "/api";
  }
  private callDatasetEndPoint = (
    what: string,
    params: Object = {}
  )  => 
  {

    const url = urlcat(this.endpoint, "/dataset/" + what, params);

    return fetch(url)
      .then(function (response) {
        return response.json();
      })
      
  };

  featuresHistogram = (features: Array<string>, callback) => {
    this.callDatasetEndPoint("histogram", {
      features: features,
      align_histograms: false,
    }).then(callback)
  };
  numericalFeaturesList = ()   :  Promise<Array<string>>  => {
    return this.callDatasetEndPoint("numerical_features_list");
  };
  categoricalFeaturesList = ()   :  Promise<Array<string>>  => {
    return this.callDatasetEndPoint("categorical_features_list");
  };

  numericalFeatures = () : Promise<DataFrameInterface> => {
    return this.callDatasetEndPoint("numerical_features")
  };

  categoricalFeatures = () : Promise<DataFrameInterface> => {
    return this.callDatasetEndPoint("categorical_features")
  };
  
  
  KLDivergenceTable = (callback: (d: DataFrameInterface) => void) => {
    this.callDatasetEndPoint("feature_kl_divergence").then(callback);
  };

  basicStatistics = (callback: (d: DataFrameInterface) => void) => {
    this.callDatasetEndPoint("basic").then(callback);
  };


  correlation = (callback: (d: DataFrameInterface) => void) => {
    this.callDatasetEndPoint("correlation").then(callback);
  };
  numberOfLives = (callback: (d: number)=> void) => {
    this.callDatasetEndPoint("number_of_lives").then(callback);
  };
  getFeatureData = (feature:string, life:number) : Promise<PlotData>  => {
    return this.callDatasetEndPoint("feature_data", {life: life, feature:feature})
  }
  durationDistribution = (callback: (d:Array<Stats>) => void) => {
    this.callDatasetEndPoint("duration_distribution").then(callback)
  }
  samplingRate = (callback: (d:BoxPlot) => void, unit:string) => {
    this.callDatasetEndPoint('sampling_rate', {'unit': unit}).then(callback)
  }
  getDistributionData = (feature:string, life:number, nbins:number, colorizedBy:string) :  Promise<PlotData>  => {
    return this.callDatasetEndPoint("distribution_data",  {life: life, feature:feature, nbins: nbins, colorized_by: colorizedBy});
  }

  getMissingValues = () : Promise<DataFrameInterface> => {
    return this.callDatasetEndPoint('missing_values')
  }
    
}

export const DatasetAPIContext = React.createContext(new DatasetAPI("localhost", 0));
export function useAPI() {
  const context = useContext(DatasetAPIContext);
  if (context === undefined) {
    throw new Error("Context must be used within a Provider");
  }
  return context;
} 

