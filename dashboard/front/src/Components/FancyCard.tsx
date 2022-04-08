import { Box, Card, CircularProgress, Grid, Paper, styled, Typography } from "@mui/material";
import React from "react";
import { ReactNode } from "react";

interface FancyCardProps {
    title: string
    children: ReactNode;

}
export const CardContent = styled(Typography)({
    textAlign: 'center',
    padding: '2rem',


});
const CardTitle = styled(Typography)({
    textAlign: 'center',
    borderRadius: 'calc(.35rem - 1px) calc(.35rem - 1px) 0 0',
    backgroundColor: '#f8f9fc',
    borderBottom: '1px solid #e3e6f0',
    padding: '1rem',
    color: '#4e73df',
    fontWeight: 700,
    paddingLeft: '2rem',
    paddingRight: '2rem'

})
const  FancyCard = (props: FancyCardProps) => {
    return (<Paper sx={{
        borderRadius: '.35rem',
        boxShadow: '0 .15rem 1.75rem 0 rgba(58,59,69,.15)'
    }}>
        <CardTitle variant="h5" > {props.title} </CardTitle>
        {props.children}

    </Paper>
    )
}
export default FancyCard;
