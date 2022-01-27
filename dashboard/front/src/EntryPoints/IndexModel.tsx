import React from "react";
import ReactDOM from "react-dom";
import { ModelAPI, ModelAPIContext } from "../Model/API";

ReactDOM.render(
    <ModelAPIContext.Provider value={new ModelAPI("http://localhost", 7575)}>
      <div>
        as
      </div>
  
    </ModelAPIContext.Provider>,
    document.getElementById("root")
  );