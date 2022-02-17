import { DataGrid, GridCallbackDetails, GridColumns, GridRenderCellParams, GridRowParams, GridToolbar, MuiEvent } from "@mui/x-data-grid";
import React, { useState, useEffect } from "react";
import LoadableComponent from "./LoadableComponent";


import { isEmpty } from "../utils";
import { DataFrameInterface } from "../Dataset/Network/Responses";



interface DataFrameProps {
  data: DataFrameInterface;
  show_index: boolean
  pagination: boolean
  pageSize: number
  selectedRowCallback?: (row: Object) => void
  renderCells?: { [column: string]: ((params: GridRenderCellParams) => React.ReactNode) }
  valid_columns_names?: Array<string>
}

const DefaultDataFrameProps = {
  show_index: true,
  pagination: true

};
export function DataFrame(props: DataFrameProps) {
  const { pageSize, renderCells = {}, valid_columns_names = [] } = props;
  if (isEmpty(props.data)) {
    return null;
  }
  const effective_columns: GridColumns = props.data.schema.fields.filter((elem) => {
    if (props.show_index) {
      return true
    } else {
      return elem.name != 'index'
    }
  })
    .filter((elem) => {
      if (valid_columns_names.length ==0) {
        return true
      } else {
        return valid_columns_names.includes(elem.name)
      }
    })
    .map((elem) => {
      const column_data = {
        Title: elem.name,
        field: elem.name,
        flex: 1,
      }
      if (elem.name in renderCells) {
        column_data['renderCell'] = renderCells[elem.name]
      }
      return column_data
    })

  const data = props.data.data.map((elem, i) => {
    const new_elem = Object.assign({}, elem);
    /*if (props.show_index == false) {
      delete new_elem['index'];
    }*/
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
    components={{
      Toolbar: GridToolbar,
    }}
    disableExtendRowFullWidth={true}
    columns={effective_columns}
    hideFooter={!props.pagination}
    disableSelectionOnClick
    autoPageSize={true}
    pageSize={pageSize}
    autoHeight={true}
    onRowClick={onRowClick}
    rowHeight={32}
  />
  )
}


DataFrame.defaultProps = DefaultDataFrameProps
const LoadableDataFrame = LoadableComponent<DataFrameInterface>(DataFrame)
export default LoadableDataFrame