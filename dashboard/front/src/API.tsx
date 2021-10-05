import React from 'react';
import urlcat from 'urlcat';
import { DataFrameInterface } from './utils';

interface KLDivergenceTableRow {
    feature: string
    mean_divergence: Number
}
export class API {
    url: string;
    port: Number;
    endpoint: string;

    constructor(url: string, port: Number) {
        this.url = url
        this.port = port
        this.endpoint = url + ':' + port + '/api'
    }
    featuresHistogram(features: Array<string>, callback) {

        const url = urlcat(this.endpoint, '/dataset/histogram', { features: features, align_histograms: false })
    
        fetch(url)
            .then(function (response) {
                return response.json();
            })
            .then(function (response: Array<string>) {

                callback(response);
            })
            .catch(function (data) {    
                console.log(data)
            });

    }
    numericalFeatures(callback: (a: Array<string>) => void) {
        fetch(this.endpoint + '/dataset/' + 'numerical_features')
            .then(function (response) {
                return response.json();
            })
            .then(function (response: Array<string>) {

                callback(response);
            })
            .catch(function (data) {
                console.log(data)
            });

    }
    KLDivergenceTable(callback: (d:DataFrameInterface)=> void) {

        fetch(this.endpoint + '/dataset/' + 'feature_kl_divergence')
        .then(function (response) {
            return response.json();
        })
        .then(function (response: DataFrameInterface) {
            callback(response);
        })
        .catch(function (data) {

            callback(data);
        });
    }


}

export const APIContext = React.createContext(new API('localhost', 0));
