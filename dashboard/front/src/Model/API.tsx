import React  from 'react';

export class ModelAPI {
    url: string;
    port: Number;
    endpoint: string;
  
    constructor(url: string, port: Number) {
      this.url = url;
      this.port = port;
      this.endpoint = url + ":" + port + "/api";
    }
}
export const ModelAPIContext = React.createContext(new ModelAPI("localhost", 0));