import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import colorlover as cl                    # Get colors from colorscales
import data_utils as du                    # Data science and machine learning relevant methods

# Color palette
colors = dict(
    gray_background='#282828',
    header_font_color='white',
    body_font_color='#ADAFAE',
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
    # Chosen dataset
    html.Div(id='dataset_name_div', children='eICU', hidden=True),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='dataset_dropdown',
                options=[
                    dict(label='eICU', value='eICU'),
                    dict(label='ALS', value='ALS')
                ],
                placeholder='Choose a dataset',
                searchable=False,
                persistence=True,
                persistence_type='session',
                style=dict(
                    color=colors['black'],
                    backgroundColor=colors['black'],
                    textColor=colors['body_font_color'],
                    fontColor=colors['body_font_color']
                )
            ),
        width=6),
        dbc.Col(
            dcc.Dropdown(
                id='model_dropdown',
                options=[
                    dict(label='RNN', value='RNN'),
                    dict(label='LSTM', value='LSTM'),
                    dict(label='TLSTM', value='TLSTM'),
                    dict(label='MF1-LSTM', value='MF1-LSTM'),
                    dict(label='MF2-LSTM', value='MF2-LSTM')
                ],
                placeholder='Choose a model',
                searchable=False,
                persistence=True,
                persistence_type='session',
                style=dict(
                    color=colors['black'],
                    backgroundColor=colors['black'],
                    textColor=colors['body_font_color'],
                    fontColor=colors['body_font_color']
                )
            ),
        width=6)
    ],
    style=dict(
        marginTop='1em',
        marginLeft='2em',
        marginRight='2em',
        textAlign='center'
    )),
    html.H5('eICU mortality prediction with LSTM model',
            id='main_title',
            style=dict(
                marginTop='0.5em',
                marginLeft='2em',
                marginRight='2em',
                textAlign='center'
            )
    ),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardBody([
                    html.H5('Performance', className='card-title'),
                    dbc.Row([
                        dbc.Col(html.Div('Accuracy'), width=6),
                        dbc.Col(du.visualization.indicator_plot(85, type='bullet', background_color=colors['gray_background'],
                                                                dash_id='accuracy_indicator',
                                                                font_color=colors['body_font_color'],
                                                                suffix='%', output_type='dash',
                                                                dash_height='1em'),
                                width=6),
                        ],
                        style=dict(
                            height='2em',
                            marginTop='1em'
                        )
                    ),
                    dbc.Row([
                        dbc.Col(html.Div('AUC'), width=6),
                        dbc.Col(du.visualization.indicator_plot(91, type='bullet', background_color=colors['gray_background'],
                                                                dash_id='accuracy_indicator', font_color=colors['body_font_color'],
                                                                prefix='0.', output_type='dash',
                                                                dash_height='1em'),
                                width=6),
                        ],
                        style=dict(
                            height='2em',
                            marginTop='1em'
                        )
                    ),
                    dbc.Row([
                        dbc.Col(html.Div('F1'), width=6),
                        dbc.Col(du.visualization.indicator_plot(60, type='bullet', background_color=colors['gray_background'],
                                                                dash_id='accuracy_indicator', font_color=colors['body_font_color'],
                                                                prefix='0.', output_type='dash',
                                                                dash_height='1em'),
                                width=6),
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
            dbc.CardBody([
                    html.H5('Dataset overview', className='card-title'),
                    dbc.Button('Expand', className='mt-auto', href='/dataset-overview'),
            ])
        ], style=dict(height='20em')),
    ], style=dict(margin='2em')),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardBody([
                    html.H5('Detailed analysis', className='card-title'),
                    dbc.Button('Expand', className='mt-auto', href='/detailed-analysis'),
            ])
        ], style=dict(height='20em')),
        dbc.Card([
            dbc.CardBody([
                    html.H5('Feature importance', className='card-title'),
                    dcc.Graph(id='feature_importance_preview',
                              config=dict(
                                displayModeBar=False
                              ),
                              style=dict(
                                height='10em',
                                marginBottom='1em'
                              )
                    ),
                    dbc.Button('Expand', className='mt-auto', href='/feature-importance'),
            ])
        ], style=dict(height='20em'))
    ], style=dict(
           marginLeft='2em',
           marginRight='2em',
           marginBottom='2m'
       )
    )
])

performance_layout = html.Div([
    # Chosen dataset
    html.Div(id='dataset_name_div', children='eICU', hidden=True),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    html.Div([
        dcc.Dropdown(
            id='model_dropdown',
            options=[
                dict(label='RNN', value='RNN'),
                dict(label='LSTM', value='LSTM'),
                dict(label='TLSTM', value='TLSTM'),
                dict(label='MF1-LSTM', value='MF1-LSTM'),
                dict(label='MF2-LSTM', value='MF2-LSTM'),
            ],
            placeholder='Choose a model',
            searchable=False,
            persistence=True,
            persistence_type='session',
            style=dict(
                color=colors['black'],
                backgroundColor=colors['black'],
                textColor=colors['body_font_color'],
                fontColor=colors['body_font_color']
            )
        ),
    ],
    style=dict(
        marginTop='1em',
        marginLeft='2em',
        marginRight='2em'
    )),
    html.H5('Performance',
            style=dict(
                marginTop='0.5em',
                marginLeft='2em',
                marginRight='2em',
                textAlign='center'
            )
    ),
    dbc.Card([
        dbc.CardBody([
                html.H5('MF1-LSTM',
                        id='model_perf_header',
                        className='card-title',
                        style=dict(color=colors['header_font_color']))
        ])
    ], color='primary', style=dict(height='5em', margin='2em')),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardHeader('Model description'),
            dbc.CardBody([
                    html.P(
                        '''LSTM-based recurrent model that accounts for time
                        variation by scaling the forget gate's output according
                        to the difference between consecutive samples\'
                        timestamps.''',
                        className='card-text',
                    ),
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                    html.H5('Test AUC', className='card-title', style=dict(color=colors['header_font_color'])),
                    du.visualization.indicator_plot(91, type='gauge', background_color=colors['gray_background'],
                                                    dash_id='perf_auc_gauge',
                                                    font_color=colors['header_font_color'],
                                                    font_size=20,
                                                    prefix='0.', output_type='dash',
                                                    dash_height='10em')
            ])
        ])
    ], style=dict(margin='2em')),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardHeader('Model parameters'),
            dbc.CardBody([
                    dbc.Table([
                        html.Tr([html.Td('n_hidden'), html.Td(100)]),
                        html.Tr([html.Td('n_rnn_layers'), html.Td(3)]),
                        html.Tr([html.Td('embedding_dim'), html.Td(str([1, 2, 3]))])
                    ], style=dict(color=colors['body_font_color']))
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                    dbc.Table([
                        html.Thead(html.Tr([html.Th('Metric'), html.Th('Train'), html.Th('Val'), html.Th('Test')])),
                        html.Tr([
                            html.Td('AUC', style=dict(width='10%')),
                            html.Td(du.visualization.indicator_plot(96, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    prefix='0.', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                            html.Td(du.visualization.indicator_plot(90, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    prefix='0.', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                            html.Td(du.visualization.indicator_plot(91, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    prefix='0.', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                        ]),
                        html.Tr([
                            html.Td('F1', style=dict(width='10%')),
                            html.Td(du.visualization.indicator_plot(82, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    prefix='0.', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                            html.Td(du.visualization.indicator_plot(30, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    prefix='0.', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                            html.Td(du.visualization.indicator_plot(60, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    prefix='0.', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                        ]),
                        html.Tr([
                            html.Td('Acc', style=dict(width='10%')),
                            html.Td(du.visualization.indicator_plot(97, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    suffix='%', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                            html.Td(du.visualization.indicator_plot(88, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    suffix='%', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                            html.Td(du.visualization.indicator_plot(85, type='bullet', background_color=colors['gray_background'],
                                                                    dash_id='accuracy_indicator',
                                                                    font_color=colors['body_font_color'],
                                                                    suffix='%', output_type='dash',
                                                                    dash_height='1em'), style=dict(width='22.5%')),
                        ]),
                    ], style=dict(color=colors['body_font_color']))
            ])
        ])
    ], style=dict(
           marginLeft='2em',
           marginRight='2em',
           marginBottom='2m'
       )
    )
])


# [TODO]
dataset_overview_layout = html.Div([
    # Chosen dataset
    html.Div(id='dataset_name_div', children='eICU', hidden=True),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    html.Div([
        dcc.Dropdown(
            id='dataset_dropdown',
            options=[
                dict(label='eICU', value='eICU'),
                dict(label='ALS', value='ALS'),
            ],
            placeholder='Choose a dataset',
            searchable=False,
            persistence=True,
            persistence_type='session',
            style=dict(
                color=colors['black'],
                backgroundColor=colors['black'],
                textColor=colors['body_font_color'],
                fontColor=colors['body_font_color']
            ),
        ),
    ],
    style=dict(
        marginTop='1em',
        marginLeft='2em',
        marginRight='2em'
    )),
    html.H5('Dataset overview',
            style=dict(
                marginTop='0.5em',
                marginLeft='2em',
                marginRight='2em',
                textAlign='center'
            )
    ),
    dbc.Card([
        dbc.CardBody([
                html.H5('eICU',
                        id='dataset_ovrvw_header',
                        className='card-title',
                        style=dict(color=colors['header_font_color']))
        ])
    ], color='primary', style=dict(height='5em', margin='2em')),
    dbc.Tabs([
        dbc.Tab(html.Div('Hello world!'), label='Size'),
        dbc.Tab(html.Div('Hello world!'), label='Demographics'),
        dbc.Tab(html.Div('Hello world!'), label='Hospital')
    ])
])

# [TODO]
feat_import_layout = html.Div([
    # Chosen dataset
    html.Div(id='dataset_name_div', children='eICU', hidden=True),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='dataset_dropdown',
                options=[
                    dict(label='eICU', value='eICU'),
                    dict(label='ALS', value='ALS')
                ],
                placeholder='Choose a dataset',
                searchable=False,
                persistence=True,
                persistence_type='session',
                style=dict(
                    color=colors['black'],
                    backgroundColor=colors['black'],
                    textColor=colors['body_font_color'],
                    fontColor=colors['body_font_color']
                )
            ),
        width=6),
        dbc.Col(
            dcc.Dropdown(
                id='model_dropdown',
                options=[
                    dict(label='RNN', value='RNN'),
                    dict(label='LSTM', value='LSTM'),
                    dict(label='TLSTM', value='TLSTM'),
                    dict(label='MF1-LSTM', value='MF1-LSTM'),
                    dict(label='MF2-LSTM', value='MF2-LSTM')
                ],
                placeholder='Choose a model',
                searchable=False,
                persistence=True,
                persistence_type='session',
                style=dict(
                    color=colors['black'],
                    backgroundColor=colors['black'],
                    textColor=colors['body_font_color'],
                    fontColor=colors['body_font_color']
                )
            ),
        width=6)
    ],
    style=dict(
        marginTop='1em',
        marginLeft='2em',
        marginRight='2em',
        textAlign='center'
    )),
    html.H5('Feature importance',
            style=dict(
                marginTop='0.5em',
                marginLeft='2em',
                marginRight='2em',
                textAlign='center'
            )
    ),
    html.P(
        'Select how the data is filtered:',
        style=dict(
            marginTop='1em',
            marginLeft='2em',
            marginRight='2em'
        )),
    html.Div([
        dcc.Dropdown(
            id='feature_importance_dropdown',
            # [TODO] Add options dynamically, according to the dataset's categorical features
            options=[
                dict(label='All', value='All'),
                dict(label='Sex', value='Sex'),
                dict(label='Age', value='Age'),
                dict(label='Diagnostic', value='Diagnostic'),
                dict(label='Treatment', value='Treatment')
            ],
            placeholder='Choose how to filter the data',
            searchable=False,
            persistence=True,
            persistence_type='session',
            multi=True,
            style=dict(
                color=colors['black'],
                backgroundColor=colors['black'],
                textColor=colors['body_font_color'],
                fontColor=colors['body_font_color']
            )
        ),
    ],
    style=dict(
        marginTop='0.5em',
        marginLeft='2em',
        marginRight='2em'
    )),
    # [TODO] Create a CardColumns that dynamically outputs cards with feature
    # importance for data filtered by the selected parameters
    dbc.CardColumns(id='feature_importance_cards', children=[])
])

# [TODO]
# detail_analysis_layout =
