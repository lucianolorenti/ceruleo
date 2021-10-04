import { AppBar, Container } from '@material-ui/core';
import MenuIcon from '@mui/icons-material/Menu';
import Box from '@mui/material/Box';
import CssBaseline from '@mui/material/CssBaseline';
import FormControl from '@mui/material/FormControl';
import Grid from '@mui/material/Grid';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import Select from '@mui/material/Select';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import React from 'react';
import ReactDOM from 'react-dom';
import { API, APIContext } from './API';
import FeatureDistribution from './feature_distribution';
import ListItemText from '@mui/material/ListItemText';

import Checkbox from '@mui/material/Checkbox';




interface DashboardProps {

}
interface FeatureSelectorProps {
  setCurrentFeatures: (a: Array<string>) => void
  features: Array<string>
  api: API
}
function FeatureSelector(props: FeatureSelectorProps) {
  const [featureList, setFeatureList] = React.useState<Array<String>>(null);
  if (featureList == null) {
    props.api.numerical_features(setFeatureList)
  }
  const handleChange = (event) => {    
    const value = event.target.value;
    console.log(value)
    props.setCurrentFeatures(
      typeof value === 'string' ? value.split(',') : value,
    );
  };


  return (
    <FormControl fullWidth>
      <InputLabel id="select-features">Features to visualize</InputLabel>
      <Select
        value={props.features}
        label="Features to visualize"
        multiple
        renderValue={(selected) => selected.join(', ')}
        onChange={handleChange}
      >
        {featureList?.map((elem: string, id: number) => {
          return <MenuItem value={elem} key={id}>
            <Checkbox checked={props.features.indexOf(elem) > -1} />
            <ListItemText primary={elem} />
          </MenuItem>

        })}

      </Select>
    </FormControl>
  )
}
function Dashboard(props: DashboardProps) {
  const [features, setCurrentFeatures] = React.useState<Array<string>>([]);
  const [open, setOpen] = React.useState<boolean>(true);
  const toggleDrawer = () => {
    setOpen(!open);
  };

  return <div>
    <APIContext.Consumer>
      {api => (
        <Box sx={{ display: 'flex' }}>
          <CssBaseline />
          <AppBar  >
            <Toolbar
              sx={{
                pr: '24px', // keep right padding when drawer closed
              }}
            >
              <IconButton
                edge="start"
                color="inherit"
                aria-label="open drawer"
                onClick={toggleDrawer}
                sx={{
                  marginRight: '36px',
                  ...(open && { display: 'none' }),
                }}
              >
                <MenuIcon />
              </IconButton>
              <Typography
                component="h1"
                variant="h6"
                color="inherit"
                noWrap
                sx={{ flexGrow: 1 }}
              >
                Dashboard
              </Typography>

            </Toolbar>
          </AppBar>
          <Box
            component="main"
            sx={{
              backgroundColor: (theme) =>
                theme.palette.mode === 'light'
                  ? theme.palette.grey[100]
                  : theme.palette.grey[900],
              flexGrow: 1,
              height: '100vh',
              overflow: 'auto',
            }}
          >
            <Toolbar />
            <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
              <Grid container spacing={3}>
        
                <Grid item xs={12} md={12} lg={12}>
                  <Paper
                    sx={{
                      p: 2,
                      display: 'flex',
                      flexDirection: 'column',
                      height: 600,
                    }}
                  >
                    <FeatureSelector api={api} features={features} setCurrentFeatures={setCurrentFeatures} />
                    <FeatureDistribution api={api} features={features} />
                  </Paper>
                </Grid>
       

                {/* Recent Orders */}
                <Grid item xs={12}>
                  <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>

                  </Paper>
                </Grid>
              </Grid>

            </Container>
          </Box>


        </Box>
      )}
    </APIContext.Consumer>
  </div>;
}

ReactDOM.render(
  <APIContext.Provider value={new API("http://localhost", 7575)} >
    <Dashboard />
  </APIContext.Provider>,
  document.getElementById('root')
);
