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
import { DatasetAPI, DatasetAPIContext } from "../Dataset/API"
import BasicStatistics from '../Dataset/BasicStatistics';
import DistributionAnalysis from '../Dataset/DistributionAnalysis';
import Correlation from '../Dataset/Correlation';
import Duration from '../Dataset/Duration'
import { Container, createMuiTheme, createTheme, ThemeProvider } from '@mui/material';
import NumericalFeatures from '../Dataset/NumericalFeatures';
import CategoricalFeatures from '../Dataset/CategoricalFeatures';
import MissingValues from '../Dataset/MissingValues';


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
  const [currentSection, setCurrentSection] = React.useState<Sections>(
    Sections.BasicStatistics
  );
  const mainComponent = (section: Sections, api: DatasetAPI) => {
    switch (section as Sections) {
      case Sections.BasicStatistics:
        return <BasicStatistics api={api} />
      case Sections.FeatureDistribution:
        return <DistributionAnalysis api={api} />
      case Sections.Correlations:
        return <Correlation api={api} />
      case Sections.Durations:
        return <Duration api={api} />
      case Sections.NumericalFeatures:
        return <NumericalFeatures  api={api}  />
      case Sections.CategoricalFeatures:
        return <CategoricalFeatures api={api}  />
      case Sections.MissingValues:
        return <MissingValues api={api} />
      default:
        return <div> aaaqq</div>
    }
  };
  return (
    <ThemeProvider theme={theme}>
    <Box sx={{ display: 'flex', backgroundColor: '#f8f9fc' } }>
      <CssBaseline />
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            RUL - PM | {currentSection}
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
              <ListItem button key={Sections[key]} onClick={(event)=> setCurrentSection(Sections[key])}>
                <ListItemIcon>
                  {index % 2 === 0 ? <InboxIcon /> : <MailIcon />}
                </ListItemIcon>
                <ListItemText primary={Sections[key]} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: '3em' }}>
        <Toolbar />
     
        <DatasetAPIContext.Consumer>
       
          {(api) => mainComponent(currentSection, api)}
        
        </DatasetAPIContext.Consumer>
    
      </Box>
    </Box>
    </ThemeProvider>
  );
}



ReactDOM.render(
  <DatasetAPIContext.Provider value={new DatasetAPI("http://localhost", 7575)}>
    <Dashboard />

  </DatasetAPIContext.Provider>,
  document.getElementById("root")
);