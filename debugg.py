"""
A simple app demonstrating how to manually construct a navbar with a customised
layout using the Navbar component and the supporting Nav, NavItem, NavLink,
NavbarBrand, and NavbarToggler components.
Requires dash-bootstrap-components 0.3.0 or later
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
nav_item = dbc.NavItem(dbc.NavLink("Link", href="#"))

# make a reuseable dropdown for the different examples
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Entry 1"),
        dbc.DropdownMenuItem("Entry 2"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Entry 3"),
    ],
    nav=True,
    in_navbar=True,
    label="Menu",
)

# this is the default navbar style created by the NavbarSimple component
default = dbc.NavbarSimple(
    children=[nav_item, dropdown],
    brand="Default",
    brand_href="#",
    sticky="top",
    className="mb-5",
)

# here's how you can recreate the same thing using Navbar
# (see also required callback at the end of the file)
custom_default = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Custom default", href="#"),
            dbc.NavbarToggler(id="navbar-toggler1"),
            dbc.Collapse(
                dbc.Nav(
                    [nav_item, dropdown], className="ms-auto", navbar=True
                ),
                id="navbar-collapse1",
                navbar=True,
            ),
        ]
    ),
    className="mb-5",
)

# this example that adds a logo to the navbar brand
logo = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("Simulador de Portfolio", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://www.entercapital.com.br/fundos?gclid=Cj0KCQjwsrWZBhC4ARIsAGGUJurewh14oGuAAFmJE7JiGhM52UfrYrCIZ2RAcdlnwn9s_jY1lcsYDO0aAgO7EALw_wcB",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler2", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(
                    [nav_item, dropdown],
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ],
    ),
    color="dark",
    dark=True,
    className="mb-5",
)


app.layout = html.Div(
    logo
)



if __name__ == "__main__":
    app.run_server(debug=True, port=8000)