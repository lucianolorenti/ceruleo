import { DataGrid } from "@mui/x-data-grid";
import React, { useState, useEffect } from "react";


import { isEmpty } from "./utils";

interface DataFrameField {
  name: string;
  type: string;
}

interface DataFrameSchema {
  fields: Array<DataFrameField>;
  primaryKey: Array<string>;
  pandas_version: string;
}

export interface DataFrameInterface {
  schema: DataFrameSchema;
  data: Array<Object>;
}

interface DataFrameProps {
  data: DataFrameInterface;
}

export function DataFrame(props: DataFrameProps) {
  if (isEmpty(props.data)) {
    return null;
  }
  const columns = props.data.schema.fields.map((elem) => {
    return {
      Title: elem.name,
      field: elem.name,
    };
  });
  const data = props.data.data.map((elem, i) => {
    elem["id"] = elem["index"];
    return elem;
  });
  return (<DataGrid
      rows={data}
      columns={columns}
      pageSize={5}
      rowsPerPageOptions={[5]}
      
      disableSelectionOnClick
    />
  )
}

