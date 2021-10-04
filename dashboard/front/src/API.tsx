import React from 'react';
import urlcat from 'urlcat';

export class API {
    url: string;
    port: Number;
    endpoint: string;

    constructor(url: string, port: Number) {
        this.url = url
        this.port = port
        this.endpoint = url + ':' + port + '/api'
    }
    features_histogram(features: Array<string>, callback) {

        const url = urlcat(this.endpoint, '/dataset/histogram', { features: features, align_histograms: false })
    
        fetch(url)
            .then(function (response) {
                return response.json();
            })
            .then(function (response: Array<string>) {

                callback(response);
            })
            .catch(function (data) {

                callback(data);
            });

    }
    numerical_features(callback: (a: Array<string>) => void) {
        fetch(this.endpoint + '/dataset/' + 'numerical_features')
            .then(function (response) {
                return response.json();
            })
            .then(function (response: Array<string>) {

                callback(response);
            })
            .catch(function (data) {

                callback(data);
            });

    }


}

export const APIContext = React.createContext(new API('localhost', 0));
