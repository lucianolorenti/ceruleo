import React from 'react';

export class API {
    url: string;
    port: Number;
    endpoint: string;

    constructor(url: string, port: Number) {
        this.url = url
        this.port = port
        this.endpoint = url + ':' + port + '/api'
    }
    features_histogram(feature: string) {
        return null;

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
