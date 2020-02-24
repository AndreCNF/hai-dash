import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

main_layout = html.Div([
    html.H5('eICU mortality prediction with LSTM model',
            style={'marginTop': '1em', 'marginLeft': '2em', 'marginRight': '2em', 'textAlign': 'center'}),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardHeader('Performance'),
            dbc.CardBody([
                    dbc.Button('Expand', className='mt-auto', href='/performance'),
            ])
        ], style={'minHeight': '20em'}),
        dbc.Card([
            dbc.CardHeader('Demographics'),
            dbc.CardBody([
                    dbc.Button('Expand', className='mt-auto', href='/demographics'),
            ])
        ], style={'minHeight': '20em'}),
    ], style={'margin': '2em'}),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardHeader('Feature importance'),
            dbc.CardBody([
                    dbc.Button('Expand', className='mt-auto', href='/feature-importance'),
            ])
        ], style={'minHeight': '20em'}),
        dbc.Card([
            dbc.CardHeader('Detailed analysis'),
            dbc.CardBody([
                    dbc.Button('Expand', className='mt-auto', href='/detailed-analysis'),
            ])
        ], style={'minHeight': '20em'})
    ], style={'marginLeft': '2em', 'marginRight': '2em', 'marginBottom': '2m'})
])

# [TODO]
# performance_layout =

# [TODO]
# demographics_layout =

# [TODO]
# feat_import_layout =

# [TODO]
# detail_analysis_layout =
