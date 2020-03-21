import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import colorlover as cl                    # Get colors from colorscales
import data_utils as du                    # Data science and machine learning relevant methods

colors = dict(
    gray_background='#282828',
    white='#fff',
    blue='#2A9FD6',
    indigo='#6610f2',
    purple='#6f42c1',
    pink='#e83e8c',
    red='#CC0000',
    orange='#fd7e14',
    yellow='#FF8800',
    green='#77B300',
    teal='#20c997',
    cyan='#9933CC',
    black='#000'
)

# Colors to use in the prediction probability bar plots
pred_colors = cl.scales['8']['div']['RdYlGn']

main_layout = html.Div([
    html.H5('eICU mortality prediction with LSTM model',
            style={'marginTop': '1em', 'marginLeft': '2em', 'marginRight': '2em', 'textAlign': 'center'}),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardHeader('Performance'),
            dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Div('Accuracy'), width=6),
                        dbc.Col(html.Div('85%'), width=3),
                        dbc.Col(du.visualization.bullet_indicator(85, background_color=colors['gray_background'],
                                                                  dash_id='accuracy_indicator'),
                                width=3),
                        ],
                        style=dict(
                            height='2em',
                            marginTop='1em'
                            )
                    ),
                    dbc.Row([
                        dbc.Col(html.Div('AUC'), width=6),
                        dbc.Col(html.Div('0.91'), width=3),
                        dbc.Col(du.visualization.bullet_indicator(91, background_color=colors['gray_background'],
                                                                  dash_id='accuracy_indicator'),
                                width=3),
                        ],
                        style=dict(
                            height='2em',
                            marginTop='1em'
                            )
                    ),
                    dbc.Row([
                        dbc.Col(html.Div('Weighted AUC'), width=6),
                        dbc.Col(html.Div('0.6'), width=3),
                        dbc.Col(du.visualization.bullet_indicator(60, background_color=colors['gray_background'],
                                                                  dash_id='accuracy_indicator'),
                                width=3),
                        ],
                        style=dict(
                            height='2em',
                            marginTop='1em'
                            )
                    ),
                    dbc.Button('Expand', className='mt-auto', href='/performance'),
            ])
        ], style=dict(height='20em')),
        dbc.Card([
            dbc.CardHeader('Demographics'),
            dbc.CardBody([
                    dbc.Button('Expand', className='mt-auto', href='/demographics'),
            ])
        ], style=dict(height='20em')),
    ], style={'margin': '2em'}),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardHeader('Feature importance'),
            dbc.CardBody([
                    dbc.Button('Expand', className='mt-auto', href='/feature-importance'),
            ])
        ], style=dict(height='20em')),
        dbc.Card([
            dbc.CardHeader('Detailed analysis'),
            dbc.CardBody([
                    dbc.Button('Expand', className='mt-auto', href='/detailed-analysis'),
            ])
        ], style=dict(height='20em'))
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
