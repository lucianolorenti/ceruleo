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
  private callDatasetEndPoint(
    what: string,
    callback: (d: any) => void,
    params: Object = {}
  ) {
    const url = urlcat(this.endpoint, "/dataset/" + what, params);
    fetch(this.endpoint + "/dataset/" + what)
      .then(function (response) {
        return response.json();
      })
      .then(callback)
      .catch(function (data) {
        console.log(data);
      });
  }

  featuresHistogram(features: Array<string>, callback) {
    this.callDatasetEndPoint("histogram", callback, {
      features: features,
      align_histograms: false,
    });
  }
  numericalFeatures(callback: (a: Array<string>) => void) {
    this.callDatasetEndPoint("numerical_features", callback);
  }
  KLDivergenceTable(callback: (d: DataFrameInterface) => void) {
    this.callDatasetEndPoint("feature_kl_divergence", callback);
  }

  basicStatistics(callback: (d: DataFrameInterface) => void) {
    this.callDatasetEndPoint("statistics", callback);
  }
}

export const APIContext = React.createContext(new API("localhost", 0));
