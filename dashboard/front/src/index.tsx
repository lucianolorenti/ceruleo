
import MenuIcon from "@mui/icons-material/Menu";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";

import IconButton from "@mui/material/IconButton";

import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import React from "react";
import ReactDOM from "react-dom";
import { API, APIContext } from "./API";

import ListItemText from "@mui/material/ListItemText";
import ListItem from "@mui/material/ListItem";
import ListItemIcon from "@mui/material/ListItemIcon";

import DistributionAnalysis from "./DistributionAnalysis";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import DashboardIcon from "@mui/icons-material/Dashboard";

import MuiDrawer from "@mui/material/Drawer";
import BasicStatistics from "./BasicStatistics";
import Correlation from "./Correlation";
import { AppBar, Container, Divider, List } from "@mui/material";

enum Sections {
  BasicStatistics = "Basic Statistics",
  MissingValues = "Missing Values",
  Correlations = "Correlations",
  Durations = "Durations",
  FeatureDistribution = "Feature Distribution",
}

interface DashboardProps {}

function Dashboard(props: DashboardProps) {
  const [currentSection, setCurrentSection] = React.useState<Sections>(
    Sections.BasicStatistics
  );
  const [open, setOpen] = React.useState<boolean>(true);
  const toggleDrawer = () => {
    setOpen(!open);
  };


  const mainComponent = (section: Sections, api: API) => {
    switch (section as Sections) {
      case Sections.BasicStatistics:
        return <BasicStatistics api={api} />
      case Sections.FeatureDistribution:
        return <DistributionAnalysis api={api} />
      case Sections.Correlations:
        return <Correlation api={api} />
      default:
        return <div> aaaqq</div> 
    }
  };
  console.log('A')
  const panels = [
    Sections.BasicStatistics,
    Sections.MissingValues,
    Sections.Correlations,
    Sections.FeatureDistribution,
  ];
  return (
    <div>
      <APIContext.Consumer>
        {(api) => (
          <Box sx={{ display: "flex" }}>
            <CssBaseline />
            <AppBar>
              <Toolbar
                sx={{
                  pr: "24px", // keep right padding when drawer closed
                }}
              >
                <IconButton
                  edge="start"
                  color="inherit"
                  aria-label="open drawer"
                  onClick={toggleDrawer}
                  sx={{
                    marginRight: "36px",
                    ...(open && { display: "none" }),
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
            <MuiDrawer variant="permanent" open={open}>
              <Toolbar
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  px: [1],
                }}
              >
                <IconButton onClick={toggleDrawer}>
                  <ChevronLeftIcon />
                </IconButton>
              </Toolbar>
              <Divider />
              <List>
                {Object.keys(Sections).map((section: Sections, i:number) => {
                  return (
                    <ListItem
                      key={i}
                      button
                      onClick={(event) => setCurrentSection(Sections[section])}
                    >
                      <ListItemIcon>
                        <DashboardIcon />
                      </ListItemIcon>
                      <ListItemText primary={Sections[section]} />
                    </ListItem>
                  );
                })}
              </List>
              <Divider />
            </MuiDrawer>
            <Box
              component="main"
              sx={{
                backgroundColor: (theme) =>
                  theme.palette.mode === "light"
                    ? theme.palette.grey[100]
                    : theme.palette.grey[900],
                flexGrow: 1,
                height: "100vh",
                overflow: "auto",
              }}
            >
              <Toolbar />
              <Container maxWidth="lg">
                {mainComponent(currentSection, api)}
              </Container>
            </Box>
          </Box>
        )}
      </APIContext.Consumer>
    </div>
  );
}

ReactDOM.render(
  <APIContext.Provider value={new API("http://localhost", 7575)}>
    <Dashboard />

  </APIContext.Provider>,
  document.getElementById("root")
);
