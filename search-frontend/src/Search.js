import React, { Component } from 'react';
import { Card, CardActions, CardHeader, CardMedia, CardTitle, CardText } from 'material-ui/Card'
import queryString from 'query-string';
import './Search.css'
import logo from './stanford-logo.png'
import { withRouter } from 'react-router-dom'
import SearchBar from 'material-ui-search-bar'
import Chip from 'material-ui/Chip';


class SearchInt extends Component {
    constructor() {
        super()
        this.state = {
            cards: [
                {
                    title: "Awesome Paper Title",
                    type: "Paper",
                    url: "http://bla.standford.edu/xyz/publications/awesome_paper.pdf",
                    filename: "awesome_paper.pdf",
                    tags: ["tag1", "longer tag2", "test tag", "really really long tag"],
                    entities: [
                        { name: "Hans Peter", type: "person", image: "http://linke-liste-konstanz.com/wp-content/uploads/2014/06/Hans-PeterKoch_color1.jpg" },
                        { name: "Hans Peter", type: "person", image: "http://linke-liste-konstanz.com/wp-content/uploads/2014/06/Hans-PeterKoch_color1.jpg" },
                        { name: "Hans Peter", type: "person", image: "http://linke-liste-konstanz.com/wp-content/uploads/2014/06/Hans-PeterKoch_color1.jpg" },
                        { name: "Hans Peter", type: "person", image: "http://linke-liste-konstanz.com/wp-content/uploads/2014/06/Hans-PeterKoch_color1.jpg" },
                        { name: "Hans Peter", type: "person", image: "http://linke-liste-konstanz.com/wp-content/uploads/2014/06/Hans-PeterKoch_color1.jpg" },
                        { name: "Hans Peter", type: "person", image: "http://linke-liste-konstanz.com/wp-content/uploads/2014/06/Hans-PeterKoch_color1.jpg" },
                        { name: "Hans Peter", type: "person", image: "http://linke-liste-konstanz.com/wp-content/uploads/2014/06/Hans-PeterKoch_color1.jpg" },
                        {
                            name: "Karlsruhe", type: "location", image: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Schloss_Karlsruhe_und_F%C3%A4cherstadt_2.jpg/300px-Schloss_Karlsruhe_und_F%C3%A4cherstadt_2.jpg"
                        }],
                    summary: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec mattis pretium massa. Aliquam erat volutpat. Nulla facilisi. Donec vulputate interdum sollicitudin. Nunc lacinia auctor quam sed pellentesque. Aliquam dui mauris, mattis quis lacus id, pellentesque lobortis odio."
                },
                {
                    title: "Awesome Paper Title",
                    type: "Paper",
                    url: "http://bla.standford.edu/xyz/publications/awesome_paper.pdf",
                    filename: "awesome_paper.pdf",
                    tags: ["tag1", "longer tag2", "test tag", "really really long tag"],
                    entities: [
                        { name: "Hans Peter", type: "person", image: "http://linke-liste-konstanz.com/wp-content/uploads/2014/06/Hans-PeterKoch_color1.jpg" },
                        {
                            name: "Karlsruhe", type: "location", image: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Schloss_Karlsruhe_und_F%C3%A4cherstadt_2.jpg/300px-Schloss_Karlsruhe_und_F%C3%A4cherstadt_2.jpg"
                        }],
                    summary: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec mattis pretium massa. Aliquam erat volutpat. Nulla facilisi. Donec vulputate interdum sollicitudin. Nunc lacinia auctor quam sed pellentesque. Aliquam dui mauris, mattis quis lacus id, pellentesque lobortis odio."
                }
            ],
            expandedCard: -1,
            searchInput: ""
        }

    }

    componentDidMount() {
        const query = queryString.parse(this.props.location.search).q;
        this.setState({
            ...this.state,
            searchInput: query
        })
        console.log("test")
        // fetch("")
        //     .then(response => response.json())
        //     .then(data => this.setState({ cards: data.cards }));
    }

    render() {
        const cards = this.state.cards
        return (
            <div>
                <div class="search-header-background">
                    <div class="search-header">
                        <a href="/" id="small-logo"><img src={logo} /></a>
                        <SearchBar
                            value={this.state.searchInput}
                            onChange={(value) => { this.setState({ searchInput: value }) }}
                            onRequestSearch={() => this.props.history.push('/search?q=' + this.state.searchInput)}
                            style={{ width: "100%" }}
                        />
                    </div>
                </div>
                {
                    cards.map((card, i) => (
                        <div class="search-content">
                            <Card containerStyle={{ padding: "10px" }}>
                                <div class="card-type">
                                    {card.type}
                                </div>
                                <div class="card-title-line">
                                    <span class="card-title" > {card.title} </span>
                                    <span class="card-filename">[{card.filename}]</span>
                                </div>
                                <a class="card-link" href={card.url}>
                                    {card.url}
                                </a>
                                <div class="card-text">
                                    {card.summary}
                                </div>
                                <div class="card-tag-container">
                                    {
                                        card.tags.map(tag => (
                                            <Chip className="card-tag"
                                                style={{ margin: "5px", fontSize: "100px", height: "27px" }}
                                                onClick={() => this.setState({...this.state, searchInput: tag})}
                                                labelStyle={{ fontSize: "12px", lineHeight: "28px" }}>{tag}</Chip>
                                        ))
                                    }
                                </div>
                                <div class="card-entities" style={this.state.expandedCard == i ? { maxHeight: "300px" } : { maxHeight: "0px" }}>
                                    {
                                        card.entities.map(entity => {
                                            return (
                                                <div class="card-entity">
                                                    <div class="card-entity-image-container">
                                                        <img src={entity.image}
                                                            onClick={() => this.setState({...this.state, searchInput: entity.name})}
                                                            class="card-entity-image" />
                                                    </div>
                                                    <div> {entity.name} </div>
                                                </div>
                                            )
                                        })
                                    }
                                </div>
                                <i class="card-expand-button fas fa-angle-double-down"
                                    onClick={() => this.setState({ expandedCard: this.state.expandedCard == i ? -1 : i })}
                                    style={this.state.expandedCard == i ? { transform: "rotate(180deg)" } : { transform: "rotate(0deg)" }}></i>
                            </Card>
                        </div>
                    ))
                }
            </div>
        )
    }
}

export const Search = withRouter(SearchInt)