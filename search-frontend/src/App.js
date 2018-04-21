import React, { Component } from 'react';
import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import { Home } from './Home'
import { Search } from './Search'
import './App.css';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'

class App extends Component {
  render() {
    return (
      <div className="App">
        <MuiThemeProvider>
          <Router>
            <div>
              <Route exact path="/" component={Home} />
              <Route path="/search" component={Search} />
            </div>
          </Router>
        </MuiThemeProvider>
      </div>
    );
  }
}

export default App;
