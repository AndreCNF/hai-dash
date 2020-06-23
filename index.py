import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
import callbacks
import layouts

base_layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink('Performance', href='/performance')),
            dbc.NavItem(dbc.NavLink('Dataset overview', href='/dataset-overview')),
            dbc.NavItem(dbc.NavLink('Feature importance', href='/feature-importance')),
            dbc.NavItem(dbc.NavLink('Detailed analysis', href='/detailed-analysis')),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem('More pages', header=True),
                    dbc.DropdownMenuItem('Tutorial', href='/tutorial'),
                    dbc.DropdownMenuItem('My profile', href='https://andrecnf.github.io/'),
                    dbc.DropdownMenuItem('Thesis', href='#'),
                ],
                nav=True,
                in_navbar=True,
                label='More',
            ),
        ],
        brand='HAI',
        brand_href='/',
        color='dark',
        dark=True,
        sticky=True,
    ),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

app.layout = base_layout

app.validation_layout = html.Div([
    base_layout,
    layouts.main_layout,
    layouts.performance_layout,
    layouts.dataset_overview_layout,
    layouts.detail_analysis_layout,
    layouts.feat_import_layout
])

if __name__ == '__main__':
    app.run_server(debug=True)
