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
import { API, APIContext } from "./API"
import BasicStatistics from './BasicStatistics';
import DistributionAnalysis from './DistributionAnalysis';
import Correlation from './Correlation';
import Duration from './Duration'
import { Container } from '@mui/material';
import NumericalFeatures from './NumericalFeatures';
import CategoricalFeatures from './CategoricalFeatures';
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';


const drawerWidth = 240;


enum Sections {
  BasicStatistics = "Basic Statistics",
  NumericalFeatures = "Numerical Features",
  CategoricalFeatures = "Categorical Features",
  MissingValues = "Missing Values",
  Correlations = "Correlations",
  Durations = "Durations",
  FeatureDistribution = "Feature Distribution",
}


export default function Dashboard() {
  const [currentSection, setCurrentSection] = React.useState<Sections>(
    Sections.BasicStatistics
  );
  const mainComponent = (section: Sections, api: API) => {
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
      default:
        return <div> aaaqq</div>
    }
  };
  console.log(currentSection)
  console.log(Sections[currentSection])
  return (
    <Box sx={{ display: 'flex' }}>
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
     
        <APIContext.Consumer>
       
          {(api) => mainComponent(currentSection, api)}
        
        </APIContext.Consumer>
    
      </Box>
    </Box>
  );
}
ReactDOM.render(
  <APIContext.Provider value={new API("http://localhost", 7575)}>
    <Dashboard />

  </APIContext.Provider>,
  document.getElementById("root")
);