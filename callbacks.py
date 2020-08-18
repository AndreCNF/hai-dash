import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, State, Output
from dash.exceptions import PreventUpdate
from dash import callback_context
import pandas as pd
import data_utils as du
from model_interpreter.model_interpreter import ModelInterpreter
import Models
import torch
import numpy as np
from time import time
import yaml
import inspect
import random
from flask_caching import Cache
from app import app
import layouts

# Set the random seed for reproducibility:
du.set_random_seed(42)
# Path to the directory where all the ML models are stored
models_path = 'models/'
metrics_path = 'metrics/individual_models/'
# [TODO] Set and update whether the model is from a custom type or not
is_custom = True
# Padding value used to pad the data sequences up to a maximum length
padding_value = 999999
# Time threshold to prevent updates during this seconds after clicking in a data point
clicked_thrsh = 5
# Load the model descriptions file
descriptions_stream = open(f'{models_path}model_descriptions.yml', 'r')
descriptions = yaml.load(descriptions_stream, Loader=yaml.FullLoader)

# Setup cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache/'
})

# Only reset cache after one week
TIMEOUT = 604800

# Index callback
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return layouts.main_layout
    elif pathname == '/performance':
        return layouts.performance_layout
    elif pathname == '/dataset-overview':
        return layouts.dataset_overview_layout
    elif pathname == '/feature-importance':
        return layouts.feat_import_layout
    elif pathname == '/detailed-analysis':
        return layouts.detail_analysis_layout
    else:
        return '404'

# Loading data callbacks
def load_dataset(file_name, file_path='', file_ext='.csv',
                 id_column='subject_id', ts_column='ts'):
    # Read the dataframe
    df = pd.read_csv(f'{file_path}{file_name}{file_ext}')
    df = df.drop(columns='Unnamed: 0')
    # Optimize the column data types
    df[id_column] = df[id_column].astype('uint')
    df[ts_column] = df[ts_column].astype('int')
    return df

@app.callback([Output('dataset_store', 'data'),
               Output('id_col_name_store', 'data'),
               Output('ts_col_name_store', 'data'),
               Output('label_col_name_store', 'data'),
               Output('cols_to_remove_store', 'data'),
               Output('reset_data_bttn', 'disabled')],
              [Input('dataset_name_div', 'children'),
               Input('sample_table', 'data_timestamp'),
               Input('model_store', 'data'),
               Input('reset_data_bttn', 'n_clicks')],
              [State('sample_table', 'data'),
               State('dataset_store', 'data')])
def load_dataset_callback(dataset_name, dataset_mod, model_file_name,
                          reset_clicks, new_data, df_store):
    if callback_context.triggered[0]['prop_id'].split('.')[0] != 'sample_table':
        # Loading a dataset from disk
        if dataset_name == 'ALS':
            data_file_name = f'fcul_als_with_shap_for_{model_file_name}'
            label_column = 'niv_label'
        elif dataset_name == 'Toy Example':
            data_file_name = f'toy_example_with_shap_for_{model_file_name}'
            label_column = 'label'
        else:
            raise Exception(f'ERROR: The HAI dashboarded isn\'t currently suited to load the dataset named {dataset_name}.')
        df = load_dataset(file_name=data_file_name, file_path='data/',
                          file_ext='.csv', id_column='subject_id')
        return df.to_dict('records'), 'subject_id', 'ts', label_column, [0, 1], True
    else:
        if dataset_name == 'ALS':
            label_column = 'niv_label'
        elif dataset_name == 'Toy Example':
            label_column = 'label'
        # Refreshing the data with a new edited sample
        df = apply_data_changes(new_data, df_store, id_column='subject_id',
                                ts_column='ts', label_column=label_column, model_file_name=model_file_name)
        return df.to_dict('records'), 'subject_id', 'ts', label_column, [0, 1], False

@app.callback([Output('model_store', 'data'),
               Output('model_metrics', 'data'),
               Output('model_hyperparam', 'data'),
               Output('is_custom_store', 'data')],
              [Input('model_name_div', 'children'),
               Input('dataset_name_div', 'children')])
def load_model_callback(model_name, dataset_name):
    global models_path
    global metrics_path
    # Based on the chosen dataset and model type, set a file path to the desired model
    if dataset_name == 'ALS':
        if model_name == 'Bidir LSTM, time aware':
            # Specify the model file name and class
            model_file_name = 'lstm_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3809valloss_06_07_2020_04_08'
            model_class = Models.VanillaLSTM
            is_custom = False
        elif model_name == 'Bidir LSTM, embedded, time aware':
            # Specify the model file name and class
            model_file_name = 'lstm_bidir_pre_embedded_delta_ts_90dayswindow_0.3481valloss_06_07_2020_04_15'
            model_class = Models.VanillaLSTM
            is_custom = False
        elif model_name == 'Bidir LSTM, embedded':
            # Specify the model file name and class
            model_file_name = 'lstm_bidir_pre_embedded_90dayswindow_0.2490valloss_06_07_2020_03_47'
            model_class = Models.VanillaLSTM
            is_custom = False
        elif model_name == 'LSTM':
            # Specify the model file name and class
            model_file_name = 'lstm_one_hot_encoded_90dayswindow_0.4363valloss_06_07_2020_03_28'
            model_class = Models.VanillaLSTM
            is_custom = False
        elif model_name == 'Bidir RNN, embedded, time aware':
            # Specify the model file name and class
            model_file_name = 'rnn_bidir_pre_embedded_delta_ts_90dayswindow_0.3059valloss_06_07_2020_03_10'
            model_class = Models.VanillaRNN
            is_custom = False
        elif model_name == 'RNN, embedded':
            # Specify the model file name and class
            model_file_name = 'rnn_with_embedding_90dayswindow_0.5569valloss_30_06_2020_17_04.pth'
            model_class = Models.VanillaRNN
            is_custom = False
        elif model_name == 'MF1-LSTM':
            # Specify the model file name and class
            model_file_name = 'mf1lstm_one_hot_encoded_90dayswindow_0.6009valloss_07_07_2020_03_46'
            model_class = Models.MF1LSTM
            is_custom = True
    elif dataset_name == 'Toy Example':
        # [TODO] Train and add each model for the toy example
        if model_name == 'Bidir LSTM, time aware':
            # Specify the model file name and class
            model_file_name = ''
            model_class = Models.VanillaLSTM
            is_custom = False
        elif model_name == 'Bidir LSTM, embedded, time aware':
            # Specify the model file name and class
            model_file_name = ''
            model_class = Models.VanillaLSTM
            is_custom = False
        elif model_name == 'Bidir LSTM, embedded':
            # Specify the model file name and class
            model_file_name = ''
            model_class = Models.VanillaLSTM
            is_custom = False
        elif model_name == 'LSTM':
            # Specify the model file name and class
            model_file_name = ''
            model_class = Models.VanillaLSTM
            is_custom = False
        elif model_name == 'Bidir RNN, embedded, time aware':
            # Specify the model file name and class
            model_file_name = ''
            model_class = Models.VanillaRNN
            is_custom = False
        elif model_name == 'RNN, embedded':
            # Specify the model file name and class
            model_file_name = ''
            model_class = Models.VanillaRNN
            is_custom = False
        elif model_name == 'MF1-LSTM':
            # Specify the model file name and class
            model_file_name = ''
            model_class = Models.MF1LSTM
            is_custom = True
    else:
        raise Exception(f'ERROR: The HAI dashboarded isn\'t currently suited to load the dataset named {dataset_name}.')
    # Load the metrics file
    metrics_stream = open(f'{metrics_path}{model_file_name}.yml', 'r')
    metrics = yaml.load(metrics_stream, Loader=yaml.FullLoader)
    # Register all the hyperparameters
    model = du.deep_learning.load_checkpoint(filepath=f'{models_path}{model_file_name}.pth', 
                                             ModelClass=model_class)
    model_args = inspect.getfullargspec(model.__init__).args[1:]
    hyperparams = dict([(param, str(getattr(model, param)))
                        for param in model_args])
    return (model_file_name,
            metrics,
            hyperparams,
            is_custom)

@app.callback(Output('model_description_list', 'children'),
              [Input('model_store', 'modified_timestamp')],
              [State('model_store', 'data')])
def load_model_description(model_mod, model_file_name):
    global descriptions
    description_list = list()
    # Add model type description
    if 'mf1lstm' in model_file_name:
        description_list.append(dbc.ListGroupItem(descriptions['model_type']['MF1-LSTM']))
    elif 'lstm' in model_file_name:
        description_list.append(dbc.ListGroupItem(descriptions['model_type']['LSTM']))
    elif 'rnn' in model_file_name:
        description_list.append(dbc.ListGroupItem(descriptions['model_type']['RNN']))
    # Add direction description
    if 'bidir' in model_file_name:
        description_list.append(dbc.ListGroupItem(descriptions['bidir']['is bidirectional']))
    else:
        description_list.append(dbc.ListGroupItem(descriptions['bidir']['single direction']))
    # Add embedding description
    if 'pre_embedded' in model_file_name:
        description_list.append(dbc.ListGroupItem(descriptions['embedding']['pre-embedded']))
    elif 'with_embedding' in model_file_name:
        description_list.append(dbc.ListGroupItem(descriptions['embedding']['with embedding']))
    else:
        description_list.append(dbc.ListGroupItem(descriptions['embedding']['one hot encoded']))
    # Add time variation description
    if 'delta_ts' in model_file_name:
        description_list.append(dbc.ListGroupItem(descriptions['delta_ts']['uses delta_ts']))
    else:
        description_list.append(dbc.ListGroupItem(descriptions['delta_ts']['no delta_ts']))
    return description_list

@app.callback(Output('hyperparam_table', 'children'),
              [Input('model_hyperparam', 'modified_timestamp')],
              [State('model_hyperparam', 'data')])
def load_hyperparameters(hyperparam_mod, hyperparams):
    rows = list()
    for key, val in hyperparams.items():
        # Add each table row, corresponding to each hyperparameter
        rows.append(html.Tr([html.Td(key), html.Td(val)]))
    return rows

@app.callback(Output('metrics_table', 'children'),
              [Input('model_metrics', 'modified_timestamp')],
              [State('model_metrics', 'data')])
def load_metrics(metrics_mod, metrics):
    rows = list()
    # Add the table header
    rows.append(html.Thead(html.Tr([html.Th('Metric'), html.Th('Train'), html.Th('Val'), html.Th('Test')])))
    for key, value in metrics['test'].items():
        # Get the values
        train_value = metrics['train'][key]
        val_value = metrics['val'][key]
        test_value = metrics['test'][key]
        # Configure the plot, according to the metrics type
        if key == 'accuracy' or key == 'precision' or key == 'recall':
            train_value = train_value * 100
            val_value = val_value * 100
            test_value = test_value * 100
            max_val = 100
            higher_is_better = True
            suffix = '%'
        elif key == 'loss':
            # Using a fictional maximum loss of 1.3, as usually a loss that
            # big is associated with a bad model performance
            max_val = 1.3
            higher_is_better = False
            suffix = None
        else:
            max_val = 1
            higher_is_better = True
            suffix = None
        # Add each table row, corresponding to each metric
        rows.append(html.Tr([
            html.Td(key), 
            html.Td(
                du.visualization.indicator_plot(train_value, min_val=0, max_val=max_val,
                                                type='bullet', 
                                                higher_is_better=higher_is_better,
                                                background_color=layouts.colors['gray_background'],
                                                dash_id=f'train_{key}_indicator',
                                                font_color=layouts.colors['body_font_color'],
                                                suffix=suffix, output_type='dash',
                                                dash_height='1.5em', animate=True), 
                style=dict(width='30%')
            ), 
            html.Td(
                du.visualization.indicator_plot(val_value, min_val=0, max_val=max_val,
                                                type='bullet', 
                                                higher_is_better=higher_is_better,
                                                background_color=layouts.colors['gray_background'],
                                                dash_id=f'val_{key}_indicator',
                                                font_color=layouts.colors['body_font_color'],
                                                suffix=suffix, output_type='dash',
                                                dash_height='1.5em', animate=True), 
                style=dict(width='30%')
            ), 
            html.Td(
                du.visualization.indicator_plot(test_value, min_val=0, max_val=max_val,
                                                type='bullet', 
                                                higher_is_better=higher_is_better,
                                                background_color=layouts.colors['gray_background'],
                                                dash_id=f'test_{key}_indicator',
                                                font_color=layouts.colors['body_font_color'],
                                                suffix=suffix, output_type='dash',
                                                dash_height='1.5em', animate=True), 
                style=dict(width='30%')
            )
        ]))
    return rows

# Dropdown callbacks
@app.callback(Output('dataset_name_div', 'children'),
              [Input('dataset_dropdown', 'value')])
def change_dataset(dataset):
    return dataset

@app.callback(Output('model_name_div', 'children'),
              [Input('model_dropdown', 'value')])
def change_model_name(model_name):
    return model_name

# Button callbacks
@app.callback(Output('edit_sample_bttn', 'children'),
              [Input('edit_sample_bttn', 'n_clicks')])
def update_edit_button(n_clicks):
    if n_clicks % 2 == 0:
        return 'Edit sample'
    else:
        return 'Stop editing'

@app.callback([Output('sample_table', 'columns'),
               Output('sample_table', 'data'),
               Output('sample_edit_div', 'hidden'),
               Output('sample_edit_card_title', 'children')],
              [Input('edit_sample_bttn', 'n_clicks')],
              [State('dataset_store', 'data'),
               State('id_col_name_store', 'data'),
               State('ts_col_name_store', 'data'),
               State('label_col_name_store', 'data'),
               State('instance_importance_graph', 'hoverData'),
               State('instance_importance_graph', 'clickData'),
               State('clicked_ts', 'children'),
               State('hovered_ts', 'children'),
               State('model_store', 'data')])
def update_sample_table(n_clicks, df_store, id_column, ts_column, label_column,
                        hovered_data, clicked_data, clicked_ts, hovered_ts,
                        model_file_name):
    if n_clicks % 2 == 0 or (hovered_data is None and clicked_data is None):
        # Stop editing
        return None, None, True, 'Sample from patient Y on timestamp X'
        # raise PreventUpdate
    else:
        global clicked_thrsh
        clicked_ts = int(clicked_ts)
        hovered_ts = int(hovered_ts)
        # Reconvert the dataframe to Pandas
        df = pd.DataFrame(df_store)
        # Check whether the current sample has originated from a hover or a click event
        if (hovered_ts - clicked_ts) <= clicked_thrsh:
            # Get the selected data point's subject ID and timestamp
            subject_id = int(clicked_data['points'][0]['y'])
            ts = clicked_data['points'][0]['x']
        else:
            # Get the selected data point's subject ID and timestamp
            subject_id = int(hovered_data['points'][0]['y'])
            ts = hovered_data['points'][0]['x']
        # Filter by the selected data point
        filtered_df = df.copy()
        filtered_df = filtered_df[(filtered_df[id_column] == subject_id)
                                  & (filtered_df[ts_column] == ts)]
        # Remove SHAP values and other unmodifiable columns from the table
        shap_column_names = [feature for feature in filtered_df.columns
                             if feature.endswith('_shap')]
        filtered_df.drop(columns=shap_column_names, inplace=True)
        filtered_df.drop(columns=label_column, inplace=True)
        if 'delta_ts' in filtered_df.columns:
            filtered_df.drop(columns='delta_ts', inplace=True)
        # Set the column names as a list of dictionaries, as data table requires
        columns = filtered_df.columns
        data_columns = [dict(name=column, id=column) for column in columns]
        return (data_columns, filtered_df.to_dict('records'), False,
                f'Sample from patient {subject_id} on timestamp {ts}')

# Page headers callbacks
@app.callback(Output('model_perf_header', 'children'),
              [Input('model_name_div', 'children')])
def change_performance_header(model_name):
    return model_name

@app.callback(Output('dataset_ovrvw_header', 'children'),
              [Input('dataset_name_div', 'children')])
def change_dataset_header(dataset):
    return dataset

@app.callback(Output('main_title', 'children'),
              [Input('dataset_name_div', 'children'),
               Input('model_name_div', 'children')])
def change_title(dataset_name, model_name):
    return f'{dataset_name} mortality prediction with {model_name} model'

# Detailed analysis strings
@app.callback(Output('clicked_ts', 'children'),
              [Input('instance_importance_graph', 'clickData')])
def update_clicked_ts(clicked_data):
    return str(int(time()))

@app.callback(Output('hovered_ts', 'children'),
              [Input('instance_importance_graph', 'hoverData')])
def update_hovered_ts(hovered_data):
    return str(int(time()))

@app.callback(Output('salient_features_card_title', 'children'),
              [Input('instance_importance_graph', 'hoverData'),
               Input('instance_importance_graph', 'clickData')],
              [State('clicked_ts', 'children')])
def update_patient_salient_feat_title(hovered_data, clicked_data, clicked_ts):
    global clicked_thrsh
    current_ts = time()
    clicked_ts = int(clicked_ts)
    # Check whether the trigger was the hover or click event
    if callback_context.triggered[0]['prop_id'].split('.')[1] == 'hoverData':
        if (current_ts - clicked_ts) <= clicked_thrsh:
            # Prevent the card from being updated on hover data if a 
            # data point has been clicked recently
            raise PreventUpdate
        # Get the selected data point's subject ID
        subject_id = hovered_data['points'][0]['y']
    else:
        # Get the selected data point's subject ID
        subject_id = clicked_data['points'][0]['y']
    return f'Patient {subject_id}\'s salient features'

@app.callback(Output('ts_feature_importance_card_title', 'children'),
              [Input('instance_importance_graph', 'hoverData'),
               Input('instance_importance_graph', 'clickData')],
              [State('clicked_ts', 'children')])
def update_ts_feat_import_title(hovered_data, clicked_data, clicked_ts):
    global clicked_thrsh
    current_ts = time()
    clicked_ts = int(clicked_ts)
    # Check whether the trigger was the hover or click event
    if callback_context.triggered[0]['prop_id'].split('.')[1] == 'hoverData':
        if (current_ts - clicked_ts) <= clicked_thrsh:
            # Prevent the card from being updated on hover data if a 
            # data point has been clicked recently
            raise PreventUpdate
        # Get the selected data point's timestamp
        ts = hovered_data['points'][0]['x']
    else:
        # Get the selected data point's timestamp
        ts = clicked_data['points'][0]['x']
    return f'Feature importance on ts={ts}'

@app.callback([Output('patient_outcome_text', 'children'),
               Output('patient_outcome_card', 'color')],
              [Input('instance_importance_graph', 'hoverData'),
               Input('instance_importance_graph', 'clickData')],
              [State('clicked_ts', 'children'),
               State('dataset_store', 'data'),
               State('dataset_name_div', 'children'),
               State('id_col_name_store', 'data'),
               State('ts_col_name_store', 'data'),
               State('label_col_name_store', 'data')])
def update_patient_outcome(hovered_data, clicked_data, clicked_ts,
                           df_store, dataset_name, id_column, 
                           ts_column, label_column):
    global clicked_thrsh
    current_ts = time()
    clicked_ts = int(clicked_ts)
    # Reconvert the dataframe to Pandas
    df = pd.DataFrame(df_store)
    # Check whether the trigger was the hover or click event
    if callback_context.triggered[0]['prop_id'].split('.')[1] == 'hoverData':
        if (current_ts - clicked_ts) <= clicked_thrsh:
            # Prevent the card from being updated on hover data if a 
            # data point has been clicked recently
            raise PreventUpdate
        # Get the selected data point's subject ID
        subject_id = int(hovered_data['points'][0]['y'])
    else:
        # Get the selected data point's subject ID
        subject_id = int(clicked_data['points'][0]['y'])
    # Filter by the selected data point
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df[id_column] == subject_id]
    # Find if the patient dies
    patient_dies = (filtered_df.tail(1)[label_column] == 1).values[0]
    if patient_dies == True:
        if dataset_name == 'ALS':
            outcome = 'Patient ends up using NIV'
        else:
            outcome = 'Patient dies'
        card_color = 'danger'
    else:
        if dataset_name == 'ALS':
            outcome = 'Patient doesn\'t need NIV in the end'
        else:
            outcome = 'Patient survives'
        card_color = 'success'
    return outcome, card_color

# Plotting
@app.callback([Output('test_accuracy_indicator_preview', 'figure'),
               Output('test_auc_indicator_preview', 'figure'),
               Output('test_f1_indicator_preview', 'figure')],
              [Input('model_metrics', 'modified_timestamp')],
              [State('model_metrics', 'data')])
def update_performance_preview(metrics_mod, metrics):
    # Get the values
    acc = metrics['test']['accuracy']
    auc = metrics['test']['AUC']
    f1 = metrics['test']['F1']
    # Accuracy plot
    acc = acc * 100
    acc_plot = du.visualization.indicator_plot(acc, min_val=0, max_val=100,
                                               type='bullet', 
                                               higher_is_better=True,
                                               background_color=layouts.colors['gray_background'],
                                               font_color=layouts.colors['body_font_color'],
                                               suffix='%', output_type='plotly')
    # AUC plot
    auc_plot = du.visualization.indicator_plot(auc, min_val=0, max_val=1,
                                               type='bullet', 
                                               higher_is_better=True,
                                               background_color=layouts.colors['gray_background'],
                                               font_color=layouts.colors['body_font_color'],
                                               suffix=None, output_type='plotly')
    # F1 plot
    f1_plot = du.visualization.indicator_plot(auc, min_val=0, max_val=1,
                                              type='bullet', 
                                              higher_is_better=True,
                                              background_color=layouts.colors['gray_background'],
                                              font_color=layouts.colors['body_font_color'],
                                              suffix=None, output_type='plotly')
    return acc_plot, auc_plot, f1_plot

@app.callback(Output('test_auc_gauge', 'figure'),
              [Input('model_metrics', 'modified_timestamp')],
              [State('model_metrics', 'data')])
def render_auc_gauge(metrics_mod, metrics):
    return du.visualization.indicator_plot(metrics['test']['AUC'], min_val=0, max_val=1,
                                           type='gauge', 
                                           background_color=layouts.colors['gray_background'],
                                           font_color=layouts.colors['header_font_color'],
                                           font_size=20,
                                           output_type='plotly')

def create_feat_import_plot(df, max_display=None,
                            xaxis_title='mean(|SHAP value|) (average impact on model output magnitude)'):
    # Get the SHAP values into a NumPy array and the feature names
    shap_column_names = [feature for feature in df.columns
                         if feature.endswith('_shap')]
    feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
    shap_values = df[shap_column_names].to_numpy()
    # Generate the SHAP summary plot
    figure = du.visualization.shap_summary_plot(shap_values, feature_names,
                                                max_display=max_display,
                                                background_color=layouts.colors['gray_background'],
                                                marker_color=layouts.colors['blue'],
                                                output_type='plotly',
                                                font_family='Roboto', font_size=14,
                                                font_color=layouts.colors['body_font_color'],
                                                xaxis_title=xaxis_title)
    return figure

def create_feat_import_card(df, card_title='Feature importance', max_display=None,
                            xaxis_title='mean(|SHAP value|) (average impact on model output magnitude)',
                            card_height=None, card_width=None):
    style = dict()
    if card_height is not None:
        style['height'] = card_height
    if card_width is not None:
        style['width'] = card_width
    feat_import_card = dbc.Card([
        dbc.CardBody([
                html.H5(card_title, className='card-title'),
                dcc.Graph(figure=create_feat_import_plot(df,
                                                         max_display=max_display,
                                                         xaxis_title=xaxis_title),
                          config=dict(
                            displayModeBar=False
                          )
                )
        ])
    ], style=style)
    return feat_import_card

@cache.memoize(timeout=TIMEOUT)
def update_feat_import_preview(model_name, df_store):
    # Reconvert the dataframe to Pandas
    df = pd.DataFrame(df_store)
    return create_feat_import_plot(df, max_display=3,
                                   xaxis_title='Average impact on output')

@app.callback(Output('feature_importance_preview', 'figure'),
              [Input('dataset_store', 'modified_timestamp'),
               Input('model_store', 'modified_timestamp')],
              [State('model_name_div', 'children'),
               State('dataset_store', 'data')])
def update_feat_import_preview_callback(dataset_mod, model_mod, model_name, df_store):
    return update_feat_import_preview(model_name, df_store)

def create_fltd_feat_import_cards(df, data_filter=None):
    # List that will contain all the cards that show feature importance for
    # each subset of data, filtered on the same categorical feature
    cards_list = list()
    if data_filter is None:
        # Use the full dataframe
        cards_list.append(create_feat_import_card(df, card_title='Feature importance',
                                                  xaxis_title='mean(|SHAP value|)'))
    else:
        # Filter the data on the specified filter (categorical feature)
        categ_columns = [column for column in df.columns
                         if column.startswith(data_filter)
                         and not column.endswith('_shap')]
        if len(categ_columns) == 1:
            # It's a typical, one column categorical feature
            one_hot_encoded = False
            # Get its unique categories
            categories = df[categ_columns[0]].unique()
        else:
            # It's a categorical feature that has been one hot encoded
            one_hot_encoded = True
            # Get its unique categories
            categories = categ_columns
        for categ in categories:
            # Filter the data to the current category and set the card title,
            # according to the current category
            if one_hot_encoded is False:
                filtered_df = df[df[data_filter] == categ]
                card_title = f'Feature importance on data with {data_filter} = {categ}'
            else:
                filtered_df = df[df[categ] == 1]
                card_title = f'Feature importance on data with {data_filter} = {categ.split(data_filter)[1]}'
            # Add a feature importance card
            cards_list.append(create_feat_import_card(filtered_df, card_title=card_title,
                                                      xaxis_title='mean(|SHAP value|)'))
    return cards_list

@app.callback(Output('feature_importance_cards', 'children'),
              [Input('feature_importance_dropdown', 'value')],
              [State('dataset_store', 'data')])
@cache.memoize(timeout=TIMEOUT)
def output_feat_import_page_cards(data_filter, df_store):
    # Reconvert the dataframe to Pandas
    df = pd.DataFrame(df_store)
    # List that will contain all the cards that show feature importance for
    # each subset of data
    cards_list = list()
    if not isinstance(data_filter, list):
        data_filter = [data_filter]
    filters = data_filter.copy()
    if 'All' in filters:
        # Add a card with feature importance on all, unfiltered data
        cards_list.append(create_fltd_feat_import_cards(df)[0])
        filters.remove('All')
    if len(filters) == 0:
        # No more feature importance cards to output
        return cards_list
    # Add the remaining cards, corresponding to feature importance on possibly
    # filtered data
    [cards_list.append(card) for fltr in filters
     for card in create_fltd_feat_import_cards(df, fltr)]
    return cards_list

@cache.memoize(timeout=TIMEOUT)
def update_det_analysis_preview(df_store, model_file_name, id_column):
    global models_path
    # Reconvert the dataframe to Pandas
    df = pd.DataFrame(df_store)
    # Find the model class
    if 'mf1lstm' in model_file_name:
        model_class = Models.MF1LSTM
    elif 'lstm' in model_file_name:
        model_class = Models.VanillaLSTM
    elif 'rnn' in model_file_name:
        model_class = Models.VanillaRNN
    # Load the model
    model = du.deep_learning.load_checkpoint(filepath=f'{models_path}{model_file_name}.pth', 
                                             ModelClass=model_class)
    # Guarantee that the model is in evaluation mode, so as to deactivate dropout
    model.eval()
    # Create a dataframe copy that doesn't include the feature importance columns
    column_names = [feature for feature in df.columns
                    if not feature.endswith('_shap')]
    tmp_df = df.copy()
    tmp_df = tmp_df[column_names]
    # Only load 4 sequences
    ids = list(tmp_df[id_column].unique())
    ids = random.sample(ids, 4)
    tmp_df = tmp_df[tmp_df[id_column].isin(ids)]
    # Calculate the instance importance scores (it should be fast enough; otherwise try to do it previously and integrate on the dataframe)
    interpreter = ModelInterpreter(model, tmp_df, inst_column=1, is_custom=is_custom)
    interpreter.interpret_model(instance_importance=True, feature_importance=False)
    # Get the instance importance plot
    return interpreter.instance_importance_plot(interpreter.test_data, 
                                                interpreter.inst_scores,
                                                labels=interpreter.test_labels,
                                                get_fig_obj=True,
                                                show_title=False,
                                                show_pred_prob=False,
                                                show_colorbar=False,
                                                max_seq=4,
                                                background_color=layouts.colors['gray_background'],
                                                font_family='Roboto', font_size=14,
                                                font_color=layouts.colors['body_font_color'])

@app.callback(Output('detailed_analysis_preview', 'figure'),
              [Input('dataset_store', 'modified_timestamp'),
               Input('model_store', 'modified_timestamp')],
              [State('dataset_store', 'data'),
               State('model_store', 'data'),
               State('id_col_name_store', 'data')])
def update_det_analysis_preview_callback(dataset_mod, model_mod, df_store, model_file_name,
                                         id_column):
    return update_det_analysis_preview(df_store, model_file_name, id_column)

@cache.memoize(timeout=TIMEOUT)
def update_full_inst_import(df_store, model_file_name, id_column, 
                            ts_column, label_column, is_custom):
    global models_path
    # Reconvert the dataframe to Pandas
    df = pd.DataFrame(df_store)
    # Find the model class
    if 'mf1lstm' in model_file_name:
        model_class = Models.MF1LSTM
    elif 'lstm' in model_file_name:
        model_class = Models.VanillaLSTM
    elif 'rnn' in model_file_name:
        model_class = Models.VanillaRNN
    # Load the model
    model = du.deep_learning.load_checkpoint(filepath=f'{models_path}{model_file_name}.pth', 
                                             ModelClass=model_class)
    # Guarantee that the model is in evaluation mode, so as to deactivate dropout
    model.eval()
    # Create a dataframe copy that doesn't include the feature importance columns
    shap_column_names = [feature for feature in df.columns
                         if feature.endswith('_shap')]
    column_names = list(df.columns)
    [column_names.remove(shap_column) for shap_column in shap_column_names]
    tmp_df = df.copy()
    tmp_df = tmp_df[column_names]
    # Calculate the instance importance scores (it should be fast enough; otherwise try to do it previously and integrate on the dataframe)
    interpreter = ModelInterpreter(model, tmp_df, id_column_name=id_column, inst_column_name=ts_column,
                                   label_column_name=label_column, fast_calc=True, 
                                   padding_value=padding_value, is_custom=is_custom, occlusion_wgt=0.7)
    interpreter.interpret_model(instance_importance=True, feature_importance=False)
    # Get the instance importance plot
    return interpreter.instance_importance_plot(interpreter.test_data, 
                                                interpreter.inst_scores,
                                                labels=interpreter.test_labels,
                                                get_fig_obj=True,
                                                show_title=False,
                                                show_pred_prob=False,
                                                show_colorbar=False,
                                                max_seq=10,
                                                background_color=layouts.colors['gray_background'],
                                                font_family='Roboto', font_size=14,
                                                font_color=layouts.colors['body_font_color'])

@app.callback(Output('instance_importance_graph', 'figure'),
              [Input('dataset_store', 'modified_timestamp'),
               Input('model_store', 'modified_timestamp'),
               Input('id_col_name_store', 'modified_timestamp'),
               Input('ts_col_name_store', 'modified_timestamp'),
               Input('label_col_name_store', 'modified_timestamp'),
               Input('is_custom_store', 'modified_timestamp')],
              [State('dataset_store', 'data'),
               State('model_store', 'data'),
               State('id_col_name_store', 'data'),
               State('ts_col_name_store', 'data'),
               State('label_col_name_store', 'data'),
               State('is_custom_store', 'data')])
def update_full_inst_import_callback(dataset_mod, model_mod, id_column_mod, ts_column_mod, 
                                     label_column_mod, is_custom_mod, df_store, 
                                     model_file_name, id_column, ts_column, label_column, 
                                     is_custom):
    return update_full_inst_import(df_store, model_file_name, id_column, 
                                   ts_column, label_column, is_custom)

@app.callback(Output('salient_features_list', 'children'),
              [Input('instance_importance_graph', 'hoverData'),
               Input('instance_importance_graph', 'clickData'),
               Input('dataset_store', 'modified_timestamp'),
               Input('model_name_div', 'children')],
              [State('dataset_store', 'data'),
               State('id_col_name_store', 'data'),
               State('clicked_ts', 'children'),
               State('hovered_ts', 'children')])
def update_most_salient_features(hovered_data, clicked_data, dataset_mod, 
                                 model_name, df_store, id_column, 
                                 clicked_ts, hovered_ts):
    global clicked_thrsh
    clicked_ts = int(clicked_ts)
    hovered_ts = int(hovered_ts)
    # Reconvert the dataframe to Pandas
    df = pd.DataFrame(df_store)
    # Check whether the current sample has originated from a hover or a click event
    if (hovered_ts - clicked_ts) <= clicked_thrsh:
        # Get the selected data point's subject ID
        subject_id = int(clicked_data['points'][0]['y'])
    else:
        # Get the selected data point's subject ID
        subject_id = int(hovered_data['points'][0]['y'])
    # Filter by the selected data point
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df[id_column] == subject_id]
    # Get the SHAP and feature values into separate NumPy arrays
    shap_column_names = [feature for feature in df.columns
                         if feature.endswith('_shap')]
    feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
    shap_values = filtered_df[shap_column_names].to_numpy()
    features = filtered_df[feature_names].to_numpy()
    # Get the instance importance plot
    return du.visualization.shap_salient_features(shap_values, features, feature_names,
                                                  max_display=6,
                                                  background_color=layouts.colors['gray_background'],
                                                  increasing_color='danger',
                                                  decreasing_color='primary',
                                                  font_family='Roboto', 
                                                  font_size=14,
                                                  font_color=layouts.colors['body_font_color'],
                                                  dash_height=None,
                                                  dash_width=None)

@app.callback(Output('ts_feature_importance_graph', 'figure'),
              [Input('instance_importance_graph', 'hoverData'),
               Input('instance_importance_graph', 'clickData'),
               Input('dataset_store', 'modified_timestamp'),
               Input('model_name_div', 'children')],
              [State('dataset_store', 'data'),
               State('id_col_name_store', 'data'),
               State('expected_value_store', 'data'),
               State('clicked_ts', 'children'),
               State('hovered_ts', 'children')])
def update_ts_feat_import(hovered_data, clicked_data, dataset_mod, 
                          model_name, df_store, id_column, 
                          exp_val, clicked_ts, hovered_ts):
    global clicked_thrsh
    clicked_ts = int(clicked_ts)
    hovered_ts = int(hovered_ts)
    # Reconvert the dataframe to Pandas
    df = pd.DataFrame(df_store)
    # Check whether the current sample has originated from a hover or a click event
    if (hovered_ts - clicked_ts) <= clicked_thrsh:
        # Get the selected data point's subject ID and timestamp
        subject_id = int(clicked_data['points'][0]['y'])
        ts = clicked_data['points'][0]['x']
    else:
        # Get the selected data point's subject ID and timestamp
        subject_id = int(hovered_data['points'][0]['y'])
        ts = hovered_data['points'][0]['x']
    # Filter by the selected data point
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df[id_column] == subject_id)
                              & (filtered_df['ts'] == ts)]
    filtered_df = filtered_df.squeeze()
    # Get the SHAP and feature values into separate NumPy arrays
    shap_column_names = [feature for feature in df.columns
                         if feature.endswith('_shap')]
    feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
    shap_values = filtered_df[shap_column_names].to_numpy()
    features = filtered_df[feature_names].to_numpy()
    # Get the instance importance plot
    return du.visualization.shap_waterfall_plot(exp_val, shap_values, features, feature_names,
                                                max_display=6, 
                                                background_color=layouts.colors['gray_background'],
                                                line_color=layouts.colors['body_font_color'], 
                                                increasing_color=layouts.colors['red'],
                                                decreasing_color=layouts.colors['blue'],
                                                font_family='Roboto', 
                                                font_size=14,
                                                font_color=layouts.colors['body_font_color'],
                                                output_type='plotly',
                                                dash_height=None,
                                                dash_width=None,
                                                expected_value_ind_height=0,
                                                output_ind_height=10)

@app.callback([Output('final_output_graph', 'figure'),
               Output('curr_final_output', 'data')],
              [Input('dataset_store', 'modified_timestamp'),
               Input('instance_importance_graph', 'hoverData'),
               Input('instance_importance_graph', 'clickData')],
              [State('dataset_store', 'data'),
               State('model_store', 'data'),
               State('model_name_div', 'children'),
               State('id_col_name_store', 'data'),
               State('ts_col_name_store', 'data'),
               State('clicked_ts', 'children'),
               State('hovered_ts', 'children'),
               State('curr_final_output', 'data')])
def update_final_output(dataset_mod, hovered_data, clicked_data, df_store, model_file_name, 
                        model_name, id_column, ts_column, clicked_ts, hovered_ts, prev_output):
    global is_custom
    global clicked_thrsh
    global models_path
    clicked_ts = int(clicked_ts)
    hovered_ts = int(hovered_ts)
    # Reconvert the dataframe to Pandas
    df = pd.DataFrame(df_store)
    # Find the model class
    if 'mf1lstm' in model_file_name:
        model_class = Models.MF1LSTM
    elif 'lstm' in model_file_name:
        model_class = Models.VanillaLSTM
    elif 'rnn' in model_file_name:
        model_class = Models.VanillaRNN
    # Load the model
    model = du.deep_learning.load_checkpoint(filepath=f'{models_path}{model_file_name}.pth', 
                                             ModelClass=model_class)
    # Guarantee that the model is in evaluation mode, so as to deactivate dropout
    model.eval()
    # Check whether the trigger was the hover or click event
    if (hovered_ts - clicked_ts) <= clicked_thrsh:
        # Get the selected data point's subject ID
        subject_id = int(clicked_data['points'][0]['y'])
    else:
        # Get the selected data point's subject ID
        subject_id = int(hovered_data['points'][0]['y'])
    # Filter by the selected data point
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df[id_column] == subject_id]
    # Only use the model-relevant features
    feature_names = [feature.split('_shap')[0] for feature in df.columns
                     if feature.endswith('_shap')]
    # feature_names = [id_column, ts_column] + feature_names
    filtered_df = filtered_df[feature_names]
    data = torch.from_numpy(filtered_df.values)
    # Remove unwanted columns from the data
    # data = du.deep_learning.remove_tensor_column(data, cols_to_remove, inplace=True)
    # Make the data three-dimensional
    data = data.unsqueeze(0)
    # Feedforward the data through the model
    outputs = model.forward(data)
    final_output = int(float(outputs[-1]) * 100)
    # Add a comparison with the previous output value, in case the current sample was edited
    if callback_context.triggered[0]['prop_id'] == 'dataset_store.modified_timestamp':
        delta_ref = prev_output
        show_delta = True
    else:
        delta_ref = None
        show_delta = False
    # Plot the updated final output
    output_plot = du.visualization.indicator_plot(final_output, type='bullet', 
                                                  higher_is_better=False,
                                                  show_delta=show_delta,
                                                  ref_value=delta_ref,
                                                  background_color=layouts.colors['gray_background'],
                                                  font_color=layouts.colors['header_font_color'],
                                                  font_size=24,
                                                  output_type='plotly')
    return output_plot, final_output

# Data editing
def apply_data_changes(new_data, df_store, id_column, ts_column, label_column, model_file_name):
    global is_custom
    global padding_value
    global models_path
    # Load the current data as a Pandas dataframe
    df = pd.DataFrame(df_store)
    # Load the new data as a Pandas dataframe
    new_sample_df = pd.DataFrame(new_data)
    # Optimize the column data types
    df[id_column] = df[id_column].astype('uint')
    df[ts_column] = df[ts_column].astype('int')
    new_sample_df[id_column] = new_sample_df[id_column].astype('uint')
    new_sample_df[ts_column] = new_sample_df[ts_column].astype('int')
    # Get the selected data point's subject ID and timestamp
    subject_id = int(new_sample_df[id_column].iloc[0])
    ts = int(new_sample_df[ts_column].iloc[0])
    # Filter by the selected data point
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df[id_column] == subject_id)
                              & (filtered_df[ts_column] == ts)]
    # Check if the new data really has changes compared to the stored data
    sample_columns = list(new_sample_df.columns)
    if new_sample_df.equals(filtered_df[sample_columns]):
        # No actual changes to be done to the dataframe
        raise PreventUpdate
    else:
        # Get the sequence data up to the edited sample, with the applied changes
        seq_df = df[(df[id_column] == subject_id)
                    & (df[ts_column] <= ts)]
        seq_df.loc[seq_df[ts_column] == ts, sample_columns] = new_sample_df.values
        # Find the sequence length dictionary corresponding to the current sample
        seq_len_dict = du.padding.get_sequence_length_dict(seq_df, id_column=id_column, ts_column=ts_column)
        # Convert the data into feature and label tensors (so as to be feed to the model)
        shap_column_names = [feature for feature in filtered_df.columns
                             if feature.endswith('_shap')]
        feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
        feature_names = [id_column, ts_column] + feature_names
        features = torch.from_numpy(seq_df[feature_names].to_numpy().astype(float))
        labels = torch.from_numpy(seq_df[label_column].to_numpy())
        # Make sure that the features are numeric (the edited data might come out in string format)
        # and three-dimensional
        features = features.float().unsqueeze(0)
        # Find the model class
        if 'mf1lstm' in model_file_name:
            model_class = Models.MF1LSTM
        elif 'lstm' in model_file_name:
            model_class = Models.VanillaLSTM
        elif 'rnn' in model_file_name:
            model_class = Models.VanillaRNN
        # Load the model
        model = du.deep_learning.load_checkpoint(filepath=f'{models_path}{model_file_name}.pth', 
                                                 ModelClass=model_class)
        # Guarantee that the model is in evaluation mode, so as to deactivate dropout
        model.eval()
        # Recalculate the SHAP values
        interpreter = ModelInterpreter(model, model_type='multivariate_rnn',
                                       id_column=0, inst_column=1, 
                                       fast_calc=True, SHAP_bkgnd_samples=10000,
                                       random_seed=du.random_seed, 
                                       padding_value=padding_value, is_custom=is_custom,
                                       seq_len_dict=seq_len_dict,
                                       feat_names=feature_names)
        _ = interpreter.interpret_model(test_data=features, test_labels=labels, 
                                        instance_importance=False, feature_importance='shap')
        # Join the updated SHAP values to the dataframe
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        data_n_shap = np.concatenate([features.numpy(), labels.unsqueeze(0).numpy(), interpreter.feat_scores], axis=2)
        # Update the current sample in the stored data
        df.loc[(df[id_column] == subject_id)
               & (df[ts_column] <= ts)] = data_n_shap.squeeze()
        return df

# Additional calculations
@app.callback(Output('expected_value_store', 'data'),
              [Input('dataset_store', 'modified_timestamp'),
               Input('model_store', 'modified_timestamp')],
              [State('model_store', 'data')])
def calc_exp_val(dataset_mod, model_mod, model_file_name):
    global models_path
    # Find the model class
    if 'mf1lstm' in model_file_name:
        model_class = Models.MF1LSTM
    elif 'lstm' in model_file_name:
        model_class = Models.VanillaLSTM
    elif 'rnn' in model_file_name:
        model_class = Models.VanillaRNN
    # Load the model
    model = du.deep_learning.load_checkpoint(filepath=f'{models_path}{model_file_name}.pth', 
                                             ModelClass=model_class)
    # Put the model in evaluation mode to deactivate dropout
    model.eval()
    # Create an all zeroes reference value
    ref_value = torch.zeros((1, 1, model.n_inputs))
    # Calculate the expected value by getting the output from the reference value
    exp_val = model(ref_value).item()
    return exp_val