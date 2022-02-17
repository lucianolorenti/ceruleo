
import { Title } from "@mui/icons-material";
import { CircularProgress, Paper, Typography } from "@mui/material";
import React, { useState, useEffect, useRef } from "react";


interface LoadableComponentProps<DataType> {
  title:string
  loading: boolean
  data: DataType
  [x: string]: any

}
export default function LoadableComponent<DataType>(WrappedComponent) {
  return (props:LoadableComponentProps<DataType>) => {
    const {
    title,
    loading,
    data,
    ...otherProps // Assume any other props are for the base element
  } = props
    const isMounted = useRef(null);


    if (loading) {
      return <CircularProgress />;
    }
    return (
      <>
        <Typography variant="h4">{props.title}</Typography>        
        <WrappedComponent data={data} {...otherProps} />
        
      </>
    );
  };
}
