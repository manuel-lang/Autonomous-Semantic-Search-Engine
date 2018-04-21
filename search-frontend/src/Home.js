import React, { Component } from 'react'
import { withRouter } from 'react-router-dom'
import SearchBar from 'material-ui-search-bar'
import "./Home.css"
import logo from './stanford-logo.png'

class HomeInt extends Component {
    constructor() {
        super();
        this.state = { searchInput: "" }
    }

    render() {
        const { history } = this.props
        return (
            <div class="home" >
                <img src={logo} id="big_logo" />
                <SearchBar
                    value={this.state.searchInput}
                    onChange={(value) => { this.setState({ searchInput: value }) }}
                    onRequestSearch={() => history.push('/search?q=' + this.state.searchInput)}
                    style={{ width: "600px", margin: "0 0 0 0" }}
                />
            </div>
        )
    }
}

export const Home = withRouter(HomeInt)