
import Paper from '@mui/material/Paper';

import React from 'react';

import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';

import Checkbox from '@mui/material/Checkbox';

export function zip(a:Array<any>, b:Array<any>)
{
    return a.map((k, i) => [k, b[i]])
}

export function isEmpty(obj) {
    return Object.keys(obj).length === 0;
  }

export interface DataFrameInterface 
{
    columns: Array<string>
    index: Array<string>
    data: Array<Array<any>>
}
export function DataFrame(data: DataFrameInterface) 
{
    return (
        <TableContainer component={Paper}>
        <Table sx={{ minWidth: 650 }} size="small" aria-label="a dense table">

            <TableHead>
                <TableRow>
                    {data.columns.map((elem, k) => <TableCell key={k}>{elem}</TableCell>)}
                </TableRow>
            </TableHead>
            <TableBody>
                {data.data.map((row, i) => (
                    <TableRow
                        key={i}
                        sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                    >
                        {row.map((elem, j) => <TableCell key={j} component="th" scope="row">
                            {elem}
                        </TableCell>
                        )}
                    </TableRow>
                ))}
            </TableBody>
        </Table>
    </TableContainer>
    )

}