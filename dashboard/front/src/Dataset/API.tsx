import React from "react";
import urlcat from "urlcat";
import { DataFrameInterface } from "./DataTable";
import { Stats } from '@visx/mock-data/lib/generators/genStats';

export interface Point {
  x: number
  y:number
}
export interface LineData {
  id: string
  data: Array<Point>
}

interface KLDivergenceTableRow {
  feature: string;
  mean_divergence: Number;
}
interface BoxPlotDataPoint {
  x:any
  y:number
}
export interface BoxPlotData {
    x: any
    min: number
    max: number
    q1: number
    q3: number
    median: number
}
export interface BoxPlot {
  boxplot: Array<BoxPlotData>
  outliers: Array<Array<number>>
}

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
    callback: (d: any) => void,
    params: Object = {}
  ) => {

    const url = urlcat(this.endpoint, "/dataset/" + what, params);

    fetch(url)
      .then(function (response) {
        return response.json();
      })
      .then(callback)
      .catch(function (data) {
        console.log(data);
      });
  };

  featuresHistogram = (features: Array<string>, callback) => {
    this.callDatasetEndPoint("histogram", callback, {
      features: features,
      align_histograms: false,
    });
  };
  numericalFeaturesList = (callback: (a: Array<string>) => void) => {
    this.callDatasetEndPoint("numerical_features_list", callback);
  };
  KLDivergenceTable = (callback: (d: DataFrameInterface) => void) => {
    this.callDatasetEndPoint("feature_kl_divergence", callback);
  };

  basicStatistics = (callback: (d: DataFrameInterface) => void) => {
    this.callDatasetEndPoint("basic", callback);
  };
  numericalFeatures = (callback: (d: DataFrameInterface) => void) => {
    this.callDatasetEndPoint("numerical_features", callback);
  };
  categoricalFeatures = (callback: (d: DataFrameInterface) => void) => {
    this.callDatasetEndPoint("categorical_features", callback);
  };
  correlation = (callback: (d: DataFrameInterface) => void) => {
    this.callDatasetEndPoint("correlation", callback);
  };
  numberOfLives = (callback: (d: number)=> void) => {
    this.callDatasetEndPoint("number_of_lives", callback);
  };
  getFeatureData = (feature:string, life:number, callback:(d:LineData) =>void) => {
    this.callDatasetEndPoint("feature_data", callback, {life: life, feature:feature});
  }
  durationDistribution = (callback: (d:Array<Stats>) => void) => {
    this.callDatasetEndPoint("duration_distribution", callback)
  }
  samplingRate = (callback: (d:BoxPlot) => void, unit:string) => {
    this.callDatasetEndPoint('sampling_rate', callback, {'unit': unit})
  }
    
}

export const DatasetAPIContext = React.createContext(new DatasetAPI("localhost", 0));
