import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import colorlover as cl                    # Get colors from colorscales
import data_utils as du                    # Data science and machine learning relevant methods

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
    html.H5('eICU mortality prediction with LSTM model',
            style=dict(
                marginTop='1em',
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
                                                                suffix='%', output_type='dash'),
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
                                                                prefix='0.', output_type='dash'),
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
                                                                prefix='0.', output_type='dash'),
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
                    html.H5('Demographics', className='card-title'),
                    dbc.Button('Expand', className='mt-auto', href='/demographics'),
            ])
        ], style=dict(height='20em')),
    ], style=dict(margin='2em')),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardBody([
                    html.H5('Feature importance', className='card-title'),
                    dbc.Button('Expand', className='mt-auto', href='/feature-importance'),
            ])
        ], style=dict(height='20em')),
        dbc.Card([
            dbc.CardBody([
                    html.H5('Detailed analysis', className='card-title'),
                    dbc.Button('Expand', className='mt-auto', href='/detailed-analysis'),
            ])
        ], style=dict(height='20em'))
    ], style=dict(
           marginLeft='2em',
           marginRight='2em',
           marginBottom='2m'
       )
    )
])

# [TODO]
performance_layout = html.Div([
    html.H5('Performance',
            style=dict(
                marginTop='1em',
                marginLeft='2em',
                marginRight='2em',
                textAlign='center'
            )
    ),
    dbc.CardColumns([
        dbc.Card([
            dbc.CardBody([
                    html.H5('MF1-LSTM', className='card-title', style=dict(color=colors['header_font_color']))
            ])
        ], color='primary', style=dict(height='5em')),
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
                    html.H5('Test AUC', className='card-title', style=dict(color=colors['header_font_color'])),
                    du.visualization.indicator_plot(91, type='gauge', background_color=colors['gray_background'],
                                                    dash_id='perf_auc_gauge',
                                                    font_color=colors['header_font_color'],
                                                    prefix='0.', output_type='dash',
                                                    dash_height='10em')
            ])
        ]),
        # [TODO] Fix the card's width to be able to accomodate all the indicator plots
        dbc.Card([
            dbc.CardBody([
                    # [TODO] Fix the bullet plots' number formating (I'll probably have to add a dash_width parameter to the indicator_plot method)
                    dbc.Table([
                        html.Thead(html.Tr([html.Th('Metric'), html.Th('Train'), html.Th('Val'), html.Th('Test')])),
                        html.Tr([
                            html.Td('AUC'),
                            html.Td(du.visualization.indicator_plot(96, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                            html.Td(du.visualization.indicator_plot(90, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                            html.Td(du.visualization.indicator_plot(91, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                        ]),
                        html.Tr([
                            html.Td('F1'),
                            html.Td(du.visualization.indicator_plot(82, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                            html.Td(du.visualization.indicator_plot(30, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                            html.Td(du.visualization.indicator_plot(60, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                        ]),
                        html.Tr([
                            html.Td('Acc'),
                            html.Td(du.visualization.indicator_plot(97, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                            html.Td(du.visualization.indicator_plot(88, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                            html.Td(du.visualization.indicator_plot(85, type='bullet', background_color=colors['gray_background'],
                                                            dash_id='accuracy_indicator',
                                                            font_color=colors['body_font_color'],
                                                            suffix='%', output_type='dash',
                                                            dash_height='1em')),
                        ]),
                    ], style=dict(color=colors['body_font_color']))
            ])
        ])
    ], style=dict(
           marginLeft='2em',
           marginRight='2em',
           marginBottom='2m',
           width='70em'
       )
    )
])


# [TODO]
# demographics_layout =

# [TODO]
# feat_import_layout =

# [TODO]
# detail_analysis_layout =
