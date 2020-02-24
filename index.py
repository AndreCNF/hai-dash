import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
import layouts
import callbacks

app.layout = html.Div([
    dbc.NavbarSimple(
    children=[
            dbc.NavItem(dbc.NavLink('Performance', href='/')),
            dbc.NavItem(dbc.NavLink('Demographics', href='/')),
            dbc.NavItem(dbc.NavLink('Feature importance', href='/')),
            dbc.NavItem(dbc.NavLink('Detailed analysis', href='/')),
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

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return layouts.main_layout
    elif pathname == '/performance':
        return layouts.performance_layout
    elif pathname == '/demographics':
        return layouts.demographics_layout
    elif pathname == '/feature-importance':
        return layouts.feat_import_layout
    elif pathname == '/detailed-analysis':
        return layouts.detail_analysis_layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)
