import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
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

detail_analysis_layout = html.Div([
    # Chosen dataset
    html.Div(id='dataset_name_div', children='ALS', hidden=True),
    dcc.Store(id='dataset_store', storage_type='memory'),
    dcc.Store(id='id_col_name_store', storage_type='memory'),
    dcc.Store(id='ts_col_name_store', storage_type='memory'),
    dcc.Store(id='label_col_name_store', storage_type='memory'),
    dcc.Store(id='cols_to_remove_store', storage_type='memory'),
    dcc.Store(id='total_length_store', storage_type='memory'),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    dcc.Store(id='model_store', storage_type='memory'),
    dcc.Store(id='model_metrics', storage_type='memory'),
    dcc.Store(id='model_hyperparam', storage_type='memory'),
    dcc.Store(id='is_custom_store', storage_type='memory'),
    dcc.Store(id='expected_value_store', storage_type='memory'),
    # Current final output value
    dcc.Store(id='curr_final_output', storage_type='memory'),
    # Current selected subject
    dcc.Store(id='curr_subject', storage_type='memory'),
    # The timestamp of the last time that a data point was clicked
    # or hovered in the instance importance graph
    html.Div(id='clicked_ts', children='0', hidden=True),
    html.Div(id='hovered_ts', children='0', hidden=True),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='dataset_dropdown',
                options=[
                    dict(label='ALS', value='ALS'),
                    dict(label='Toy Example', value='Toy Example')
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
                    dict(label='Bidir LSTM, time aware', value='Bidir LSTM, time aware'),
                    dict(label='Bidir LSTM, embedded, time aware', value='Bidir LSTM, embedded, time aware'),
                    dict(label='Bidir LSTM, embedded', value='Bidir LSTM, embedded'),
                    dict(label='LSTM', value='LSTM'),
                    dict(label='Bidir RNN, embedded, time aware', value='Bidir RNN, embedded, time aware'),
                    dict(label='RNN, embedded', value='RNN, embedded'),
                    dict(label='MF1-LSTM', value='MF2-LSTM')
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
    html.H5(
        'Detailed analysis',
        style=dict(
            marginTop='0.5em',
            marginLeft='2em',
            marginRight='2em',
            textAlign='center'
        )
    ),
    # [TODO] Add dynamic options to filter the data
    dbc.Card([
            dbc.CardBody([
                    html.H5('Instance importance on all patients\' time series', className='card-title'),
                    dcc.Loading(
                        id='loading_instance_importance_graph',
                        children=[
                            dcc.Graph(
                                id='instance_importance_graph',
                                config=dict(
                                    displayModeBar=False
                                ),
                                style=dict(
                                    height='24em',
                                    marginBottom='1em'
                                )
                            ),
                        ],
                        type='default',
                    )
            ])
        ], style=dict(
               height='30.5em',
               marginTop='1em',
               marginLeft='2em',
               marginRight='2em'
           )
    ),
    dbc.CardColumns(
        children=[
            dbc.Card([
                dbc.CardBody([
                        html.H5(
                            'Patient Y\'s salient features',
                            id='salient_features_card_title',
                            className='card-title'
                        ),
                        dcc.Loading(
                            id='loading_salient_features_list',
                            children=[
                                dbc.ListGroup(
                                    id='salient_features_list',
                                    children=[],
                                    style=dict(
                                        height='20em',
                                        marginBottom='1em'
                                    )
                                )
                            ],
                            type='default',
                        )
                ])
            ], style=dict(height='26em')),
            dbc.Card([
                dbc.CardBody([
                        html.H5(
                            'Feature importance on ts=X',
                            id='ts_feature_importance_card_title',
                            className='card-title'
                        ),
                        dcc.Loading(
                            id='loading_ts_feature_importance_graph',
                            children=[
                                dcc.Graph(
                                    id='ts_feature_importance_graph',
                                    config=dict(
                                        displayModeBar=False
                                    ),
                                    style=dict(
                                        height='20em',
                                        marginBottom='1em'
                                    )
                                )
                            ],
                            type='default',
                        )
                ])
            ], style=dict(height='26em')),
            dbc.Card([
                dbc.CardBody([
                        html.H5(
                            'Patient outcome',
                            id='patient_outcome_text',
                            className='card-title',
                            style=dict(color=colors['header_font_color'])
                        )
                ])
            ], id='patient_outcome_card', color='secondary', style=dict(height='5em')),
            dbc.Card([
                dbc.CardBody([
                        html.H5('Final output', className='card-title'),
                        dcc.Graph(id='final_output_graph',
                                  config=dict(
                                      displayModeBar=False
                                  ),
                                  style=dict(
                                      height='5em',
                                    #   marginBottom='1em'
                                  ),
                                  animate=True
                        ),
                ])
            ], style=dict(height='12em')),
            # [TODO] Create a part that allows editing the selected sample and see its effect;
            # this button should redirect to that part
            dbc.Button('Reset data', id='reset_data_bttn', className='mt-auto', size='lg',
                       block=True, disabled=True, style=dict(marginBottom='0.5em')),
            dbc.Button('Edit sample', id='edit_sample_bttn', className='mt-auto', size='lg', block=True),
        ],
        style=dict(
            marginTop='1em',
            marginLeft='2em',
            marginRight='2em'
        )
    ),
    html.Div(
        id='sample_edit_div',
        hidden=True,
        children=[
            dbc.Card([
                dbc.CardBody([
                        html.H5(
                            'Sample from patient Y on timestamp X', 
                            id='sample_edit_card_title',
                            className='card-title'
                        ),
                        dash_table.DataTable(
                            id='sample_table',
                            editable=True,
                            style_table=dict(overflowX='auto'),
                            style_header=dict(
                                backgroundColor=colors['black'],
                                border='1px solid grey'
                            ),
                            style_cell=dict(
                                backgroundColor=colors['gray_background'],
                                color='white',
                                border='1px solid grey'
                            ),
                        ),
                ])
            ], style=dict(
                    height='10em',
                    marginLeft='2em',
                    marginRight='2em'
                )
            ),
        ])
])

main_layout = html.Div([
    # I need to have the detailed analysis hidden here, just so Dash can be aware of its components in the main page
    html.Div(detail_analysis_layout, hidden=True),
    # Chosen dataset
    html.Div(id='dataset_name_div', children='ALS', hidden=True),
    dcc.Store(id='dataset_store', storage_type='memory'),
    dcc.Store(id='id_col_name_store', storage_type='memory'),
    dcc.Store(id='ts_col_name_store', storage_type='memory'),
    dcc.Store(id='label_col_name_store', storage_type='memory'),
    dcc.Store(id='cols_to_remove_store', storage_type='memory'),
    dcc.Store(id='total_length_store', storage_type='memory'),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    dcc.Store(id='model_store', storage_type='memory'),
    dcc.Store(id='model_metrics', storage_type='memory'),
    dcc.Store(id='model_hyperparam', storage_type='memory'),
    dcc.Store(id='is_custom_store', storage_type='memory'),
    dcc.Store(id='expected_value_store', storage_type='memory'),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='dataset_dropdown',
                options=[
                    dict(label='ALS', value='ALS'),
                    dict(label='Toy Example', value='Toy Example')
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
                    dict(label='Bidir LSTM, time aware', value='Bidir LSTM, time aware'),
                    dict(label='Bidir LSTM, embedded, time aware', value='Bidir LSTM, embedded, time aware'),
                    dict(label='Bidir LSTM, embedded', value='Bidir LSTM, embedded'),
                    dict(label='LSTM', value='LSTM'),
                    dict(label='Bidir RNN, embedded, time aware', value='Bidir RNN, embedded, time aware'),
                    dict(label='RNN, embedded', value='RNN, embedded'),
                    dict(label='MF1-LSTM', value='MF2-LSTM')
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
    html.H5(
        'ALS NIV prediction with LSTM model',
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
                        dbc.Col(
                            dcc.Graph(
                                id='test_accuracy_indicator_preview',
                                config=dict(
                                    displayModeBar=False
                                ),
                                style=dict(
                                    height='1em'
                                ),
                                animate=True
                            ),
                            width=6
                        ),
                        ],
                        style=dict(
                            height='2em',
                            marginTop='1em'
                        )
                    ),
                    dbc.Row([
                        dbc.Col(html.Div('AUC'), width=6),
                        dbc.Col(
                            dcc.Graph(
                                id='test_auc_indicator_preview',
                                config=dict(
                                    displayModeBar=False
                                ),
                                style=dict(
                                    height='1em'
                                ),
                                animate=True
                            ),
                            width=6
                        ),
                        ],
                        style=dict(
                            height='2em',
                            marginTop='1em'
                        )
                    ),
                    dbc.Row([
                        dbc.Col(html.Div('F1'), width=6),
                        dbc.Col(
                            dcc.Graph(
                                id='test_f1_indicator_preview',
                                config=dict(
                                    displayModeBar=False
                                ),
                                style=dict(
                                    height='1em'
                                ),
                                animate=True
                            ),
                            width=6
                        ),
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
                    dcc.Loading(
                        id='loading_detailed_analysis_preview',
                        children=[
                            dcc.Graph(
                                id='detailed_analysis_preview',
                                config=dict(
                                    displayModeBar=False
                                ),
                                style=dict(
                                    height='10em',
                                    marginBottom='1em'
                                )
                            ),
                        ],
                        type='default',
                    ),
                    dbc.Button('Expand', className='mt-auto', href='/detailed-analysis'),
            ])
        ], style=dict(height='20em')),
        dbc.Card([
            dbc.CardBody([
                    html.H5('Feature importance', className='card-title'),
                    dcc.Loading(
                        id='loading_feature_importance_preview',
                        children=[
                            dcc.Graph(
                                id='feature_importance_preview',
                                config=dict(
                                    displayModeBar=False
                                ),
                                style=dict(
                                    height='10em',
                                    marginBottom='1em'
                                )
                            ),
                        ],
                        type='default',
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
    html.Div(id='dataset_name_div', children='ALS', hidden=True),
    dcc.Store(id='dataset_store', storage_type='memory'),
    dcc.Store(id='id_col_name_store', storage_type='memory'),
    dcc.Store(id='ts_col_name_store', storage_type='memory'),
    dcc.Store(id='label_col_name_store', storage_type='memory'),
    dcc.Store(id='cols_to_remove_store', storage_type='memory'),
    dcc.Store(id='total_length_store', storage_type='memory'),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    dcc.Store(id='model_store', storage_type='memory'),
    dcc.Store(id='model_metrics', storage_type='memory'),
    dcc.Store(id='model_hyperparam', storage_type='memory'),
    dcc.Store(id='is_custom_store', storage_type='memory'),
    dcc.Store(id='expected_value_store', storage_type='memory'),
    html.Div([
        dcc.Dropdown(
            id='model_dropdown',
            options=[
                dict(label='Bidir LSTM, time aware', value='Bidir LSTM, time aware'),
                dict(label='Bidir LSTM, embedded, time aware', value='Bidir LSTM, embedded, time aware'),
                dict(label='Bidir LSTM, embedded', value='Bidir LSTM, embedded'),
                dict(label='LSTM', value='LSTM'),
                dict(label='Bidir RNN, embedded, time aware', value='Bidir RNN, embedded, time aware'),
                dict(label='RNN, embedded', value='RNN, embedded'),
                dict(label='MF1-LSTM', value='MF2-LSTM')
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
    html.H5(
        'Performance',
        style=dict(
            marginTop='0.5em',
            marginLeft='2em',
            marginRight='2em',
            textAlign='center'
        )
    ),
    dbc.Card([
        dbc.CardBody([
                html.H5(
                    'MF1-LSTM',
                    id='model_perf_header',
                    className='card-title',
                    style=dict(color=colors['header_font_color'])
                )
        ])
    ], color='primary', style=dict(height='5em', margin='2em')),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardHeader('Model description'),
            dbc.CardBody([
                dbc.ListGroup(
                    id='model_description_list',
                    children=[]
                ),
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                html.H5(
                    'Test AUC',
                    className='card-title',
                    style=dict(color=colors['header_font_color'])
                ),
                dcc.Graph(
                    id='test_auc_gauge',
                    config=dict(
                        displayModeBar=False
                    ),
                    style=dict(
                        height='17em'
                    ),
                    animate=True
                )
            ])
        ])
    ], style=dict(margin='2em')),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardHeader('Model parameters'),
            dbc.CardBody([dbc.Table(
                id='hyperparam_table',
                children=[
                        html.Tr([html.Td('n_hidden'), html.Td(100)]),
                        html.Tr([html.Td('n_rnn_layers'), html.Td(3)]),
                        html.Tr([html.Td('embedding_dim'), html.Td(str([1, 2, 3]))])
                ], 
                style=dict(color=colors['body_font_color']))
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                dbc.Table(
                    id='metrics_table',
                    children=[], 
                    style=dict(color=colors['body_font_color'])
                )
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
    html.Div(id='dataset_name_div', children='ALS', hidden=True),
    dcc.Store(id='dataset_store', storage_type='memory'),
    dcc.Store(id='id_col_name_store', storage_type='memory'),
    dcc.Store(id='ts_col_name_store', storage_type='memory'),
    dcc.Store(id='label_col_name_store', storage_type='memory'),
    dcc.Store(id='cols_to_remove_store', storage_type='memory'),
    dcc.Store(id='total_length_store', storage_type='memory'),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    dcc.Store(id='model_store', storage_type='memory'),
    dcc.Store(id='model_metrics', storage_type='memory'),
    dcc.Store(id='model_hyperparam', storage_type='memory'),
    dcc.Store(id='is_custom_store', storage_type='memory'),
    dcc.Store(id='expected_value_store', storage_type='memory'),
    # Feature filtering dropdown placeholders
    html.Div(children=dcc.Dropdown(id='seq_len_dist_dropdown'), hidden=True),
    html.Div(children=dcc.Dropdown(id='time_freq_dist_dropdown'), hidden=True),
    html.Div([
        dcc.Dropdown(
            id='dataset_dropdown',
            options=[
                dict(label='ALS', value='ALS'),
                dict(label='Toy Example', value='Toy Example')
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
    ],
    style=dict(
        marginTop='1em',
        marginLeft='2em',
        marginRight='2em'
    )),
    html.H5(
        'Dataset overview',
        style=dict(
            marginTop='0.5em',
            marginLeft='2em',
            marginRight='2em',
            textAlign='center'
        )
    ),
    dbc.Card([
        dbc.CardBody([
                html.H5(
                    'ALS',
                    id='dataset_ovrvw_header',
                    className='card-title',
                    style=dict(color=colors['header_font_color'])
                )
        ])
    ], color='primary', style=dict(height='5em', margin='2em')),
    dbc.Tabs([
        dbc.Tab([
            dbc.CardColumns(
                id='dataset_size_num_cards',
                children=[],
                style=dict(
                    marginTop='1em',
                    marginLeft='2em',
                    marginRight='2em'
                )
            ),
            dcc.Loading(
                id='loading_dataset_size_plot_cards',
                children=[
                    dbc.CardDeck(
                        id='dataset_size_plot_cards',
                        children=[], 
                        style=dict(
                            marginTop='1em',
                            marginLeft='2em',
                            marginRight='2em'
                        )
                    ),
                ],
                type='default',
            )
        ], label='Size'),
        dbc.Tab(
            dcc.Loading(
                id='loading_dataset_demographics_cards',
                children=[
                    dbc.CardColumns(
                        id='dataset_demographics_cards',
                        children=[],
                        style=dict(
                            marginTop='1em',
                            marginLeft='2em',
                            marginRight='2em'
                        )
                    )
                ],
                type='default',
            ), 
            label='Demographics'
        ),
        dbc.Tab(
            dcc.Loading(
                id='loading_dataset_info_cards',
                children=[
                    dbc.CardColumns(
                        id='dataset_info_cards',
                        children=[],
                        style=dict(
                            marginTop='1em',
                            marginLeft='2em',
                            marginRight='2em'
                        )
                    )
                ],
                type='default',
            ), 
            label='Additional info'
        )
    ])
])

# [TODO]
feat_import_layout = html.Div([
    # Chosen dataset
    html.Div(id='dataset_name_div', children='ALS', hidden=True),
    dcc.Store(id='dataset_store', storage_type='memory'),
    dcc.Store(id='id_col_name_store', storage_type='memory'),
    dcc.Store(id='ts_col_name_store', storage_type='memory'),
    dcc.Store(id='label_col_name_store', storage_type='memory'),
    dcc.Store(id='cols_to_remove_store', storage_type='memory'),
    dcc.Store(id='total_length_store', storage_type='memory'),
    # Chosen machine learning model
    html.Div(id='model_name_div', children='LSTM', hidden=True),
    dcc.Store(id='model_store', storage_type='memory'),
    dcc.Store(id='model_metrics', storage_type='memory'),
    dcc.Store(id='model_hyperparam', storage_type='memory'),
    dcc.Store(id='is_custom_store', storage_type='memory'),
    dcc.Store(id='expected_value_store', storage_type='memory'),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='dataset_dropdown',
                options=[
                    dict(label='ALS', value='ALS'),
                    dict(label='Toy Example', value='Toy Example')
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
                    dict(label='Bidir LSTM, time aware', value='Bidir LSTM, time aware'),
                    dict(label='Bidir LSTM, embedded, time aware', value='Bidir LSTM, embedded, time aware'),
                    dict(label='Bidir LSTM, embedded', value='Bidir LSTM, embedded'),
                    dict(label='LSTM', value='LSTM'),
                    dict(label='Bidir RNN, embedded, time aware', value='Bidir RNN, embedded, time aware'),
                    dict(label='RNN, embedded', value='RNN, embedded'),
                    dict(label='MF1-LSTM', value='MF2-LSTM')
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
    html.H5(
        'Feature importance',
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
                dict(label='Var0', value='Var0'),
                dict(label='Var4_a', value='Var4_a'),
                dict(label='Var4_b', value='Var4_b'),
                dict(label='Var4_c', value='Var4_c'),
                # dict(label='Sex', value='Sex'),
                # dict(label='Age', value='Age'),
                # dict(label='Diagnostic', value='Diagnostic'),
                # dict(label='Treatment', value='Treatment')
            ],
            placeholder='Choose how to filter the data',
            value='All',
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
    # Create a CardColumns that dynamically outputs cards with feature
    # importance for data filtered by the selected parameters
    dcc.Loading(
        id='loading_feature_importance_cards',
        children=[
            dbc.CardColumns(
                id='feature_importance_cards',
                children=[],
                style=dict(
                    marginTop='1em',
                    marginLeft='2em',
                    marginRight='2em'
                )
            )
        ],
        type='default',
    )
])

