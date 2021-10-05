import Paper from "@mui/material/Paper";

import React, { useState, useEffect } from "react";

import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import { CircularProgress } from "@material-ui/core";
import { API } from "./API";

export interface DataFrameInterface {
  columns: Array<string>;
  index: Array<string>;
  data: Array<Array<any>>;
}

interface DataFrameProps {
  table: DataFrameInterface;
}
function DataFrame(props: DataFrameProps) {
  return (
    <TableContainer component={Paper}>
      <Table sx={{ minWidth: 650 }} size="small" aria-label="a dense table">
        <TableHead>
          <TableRow>
            {props.table.columns.map((elem, k) => (
              <TableCell key={k}>{elem}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {props.table.data.map((row, i) => (
            <TableRow
              key={i}
              sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
            >
              {row.map((elem, j) => (
                <TableCell key={j} component="th" scope="row">
                  {elem}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export interface DataTableProps {
  title: string;
  fetcher: (a: (a: DataFrameInterface) => void) => void;
}
export default function DataTable(props: DataTableProps) {
  const [dataTable, setDataTable] = useState<DataFrameInterface>({
    columns: [],
    index: [],
    data: [],
  });
  const [loading, setLoading] = useState(false);
  const setDataReady = (data: DataFrameInterface) => {
    setLoading(false);
    setDataTable(data);
  };

  useEffect(() => {
    setLoading(true);
    props.fetcher(setDataReady);
  }, []);
  if (loading) {
    return <CircularProgress />;
  }

  return (
    <div>
      <h1> props.title </h1>
      <DataFrame table={dataTable} />
    </div>
  );
}
