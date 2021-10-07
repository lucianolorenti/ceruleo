import React from "react";
import urlcat from "urlcat";
import { DataFrameInterface } from "./DataTable";

interface KLDivergenceTableRow {
  feature: string;
  mean_divergence: Number;
}
export class API {
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
    console.log(url)
    fetch(this.endpoint + "/dataset/" + what)
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
  numericalFeaturesDistribution = (callback: (a: Array<string>) => void) => {
    this.callDatasetEndPoint("numerical_features_distribution", callback);
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
  getFeatureData = (feature:string, life:number, callback:(d:number[]) =>void) => {
    this.callDatasetEndPoint("feature_data", callback, {life: life, feature:feature});
  }
    
}

export const APIContext = React.createContext(new API("localhost", 0));
