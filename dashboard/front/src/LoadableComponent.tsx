import { CircularProgress } from "@material-ui/core";
import React, { useState, useEffect } from "react";


export default function LoadableComponent(title:string, fetcher: (a: (a: any) => void) => void, WrappedComponent) {
  return (props) => {
    const [data, setData] = useState<any>({});
    const [loading, setLoading] = useState(false);
    const setDataReady = (data: any) => {
      setLoading(false);
      setData(data);
    };

    useEffect(() => {
      setLoading(true);
      fetcher(setDataReady);
    }, []);
    if (loading) {
      return <CircularProgress />;
    }
    return (
      <div>
        <h1>{title}</h1>
        <WrappedComponent data={data} {...props} />
      </div>
    );
  };
}
