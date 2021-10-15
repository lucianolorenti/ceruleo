
import { CircularProgress } from "@mui/material";
import React, { useState, useEffect, useRef } from "react";


interface LoadableComponentProps {
  title:string
  fetcher: (a: (a: any) => void) => void
  [x: string]: any

}
export default function LoadableComponent(WrappedComponent) {
  return (props:LoadableComponentProps) => {
    const {
    title,
    fetcher,
    ...otherProps // Assume any other props are for the base element
  } = props
    const [data, setData] = useState<any>({});
    const [loading, setLoading] = useState(false);
    const isMounted = useRef(null);

    const setDataReady = (data: any) => {
      if (isMounted.current) {
      setLoading(false);
      setData(data);
      }
      
    
    };
   
    useEffect(() => {
      isMounted.current = true;
      setLoading(true);
      fetcher(setDataReady);
      return () => {
        
        isMounted.current = false;
      }

    }, []);
    if (loading) {
      return <CircularProgress />;
    }
    return (
      <div>
        <h1>{props.title}</h1>
        <div>
        <WrappedComponent data={data} {...otherProps} />
        </div>
      </div>
    );
  };
}
