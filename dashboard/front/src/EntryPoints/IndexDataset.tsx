import * as React from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import AppBar from '@mui/material/AppBar';
import CssBaseline from '@mui/material/CssBaseline';
import Toolbar from '@mui/material/Toolbar';
import List from '@mui/material/List';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import InboxIcon from '@mui/icons-material/MoveToInbox';
import MailIcon from '@mui/icons-material/Mail';
import ReactDOM from "react-dom";
import { DatasetAPI, DatasetAPIContext } from "../Dataset/Network/API"
import BasicStatistics from '../Dataset/BasicStatistics';
import DistributionAnalysis from '../Dataset/DistributionAnalysis';
import Correlation from '../Dataset/Correlation';
import Duration from '../Dataset/Duration'
import { Button, Container, createMuiTheme, createTheme, Link, ThemeProvider } from '@mui/material';
import NumericalFeatures from '../Dataset/NumericalFeatures';
import CategoricalFeatures from '../Dataset/CategoricalFeatures';
import MissingValues from '../Dataset/MissingValues';
import {
  BrowserRouter,
  Routes,
  Route,
  useLocation
} from "react-router-dom";
import { Link as RouterLink } from 'react-router-dom';
import { FeaturesProvider } from '../Dataset/Store/FeatureNames';
import { FeaturesDataProvider } from '../Dataset/Store/FeatureTables';
import { lightGreen } from '@mui/material/colors';

const drawerWidth = 240;


enum Sections {
  BasicStatistics = "Basic Statistics",
  MissingValues = "Missing Values",
  NumericalFeatures = "Numerical Features",
  CategoricalFeatures = "Categorical Features",

  Correlations = "Correlations",
  Durations = "Durations",
  FeatureDistribution = "Feature Distribution",
}

const theme = createTheme({
  typography: {
    fontFamily: [
      'Nunito',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif'
    ].join(','),
  }
});


export default function Dashboard() {

  const location = useLocation();

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex', backgroundColor: '#f8f9fc' }}>
        <CssBaseline />
        <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <Typography variant="h6" noWrap component="div">
              RUL - PM | {Sections[location.pathname.substr(1)]}
            </Typography>
          </Toolbar>
        </AppBar>
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto' }}>
            <List>
              {Object.keys(Sections).map((key, index) => (

                <ListItem button key={Sections[key]} >
                  <ListItemIcon>
                    {index % 2 === 0 ? <InboxIcon /> : <MailIcon />}
                  </ListItemIcon>
                  <Button to={"/" + key} component={RouterLink} >{Sections[key]}</Button>

                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>
        <Box component="main" sx={{ flexGrow: 1, p: '1em' }}>
          <Toolbar />

          <DatasetAPIContext.Consumer>
            {(api) =>


              <Routes>
                <Route path="BasicStatistics" element={<BasicStatistics  />} />
                <Route path="NumericalFeatures" element={<NumericalFeatures />} />
                <Route path="MissingValues" element={<MissingValues />} />



              </Routes>

            }

          </DatasetAPIContext.Consumer>

        </Box>
      </Box>
    </ThemeProvider>
  );
}



ReactDOM.render(
  <DatasetAPIContext.Provider value={new DatasetAPI("http://localhost", 7575)}>
    <BrowserRouter>
    <FeaturesProvider>
      <FeaturesDataProvider>
      <Dashboard />
      </FeaturesDataProvider>
     </FeaturesProvider> 
    </BrowserRouter>
  </DatasetAPIContext.Provider>,
  document.getElementById("root")
);

/*
<Route path="FeatureDistribution" element={<DistributionAnalysis api={api} />} />
                <Route path="Correlations" element={<Correlation api={api} />} />
                <Route path="Durations" element={<Duration api={api} />} />
                
                <Route path="CategoricalFeatures" element={<CategoricalFeatures api={api} />} />
                

  
*/