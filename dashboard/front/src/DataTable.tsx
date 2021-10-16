import { DataGrid, GridCallbackDetails, GridRowParams, MuiEvent } from "@mui/x-data-grid";
import React, { useState, useEffect } from "react";
import LoadableComponent from "./LoadableComponent";


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
  show_index: boolean
  pagination: boolean
  selectedRowCallback?: (row: Object) => void
}

const DefaultDataFrameProps = {
  show_index: true,
  pagination: true
};
export function DataFrame(props: DataFrameProps) {
  if (isEmpty(props.data)) {
    return null;
  }
  const columns = props.data.schema.fields.filter((elem) => {
    if (props.show_index) {
      return true
    } else {
      return elem.name != 'index'
    }
  })
    .map((elem) => {
      return {
        Title: elem.name,
        field: elem.name,
        flex: 1
      };
    })

  const data = props.data.data.map((elem, i) => {
    const new_elem = Object.assign({}, elem);
    if (props.show_index == false) {
      delete new_elem['index'];
    }
    new_elem["id"] = props.data.schema.primaryKey.map((pk) => new_elem[pk]).join('_');
    return new_elem;
  });
  const onRowClick = (params: GridRowParams, event: MuiEvent<React.MouseEvent>, details: GridCallbackDetails) => {
    if (props.selectedRowCallback != null) {
      props.selectedRowCallback(params.row)
    }

  }

  return (<DataGrid
    rows={data}
    autoHeight={true}
    disableExtendRowFullWidth={true}
    columns={columns}
    hideFooter={!props.pagination}
    disableSelectionOnClick
    pageSize={7}
    onRowClick={onRowClick}
    rowHeight={32}
  />
  )
}


DataFrame.defaultProps = DefaultDataFrameProps
const LoadableDataFrame = LoadableComponent(DataFrame)
export default LoadableDataFrame