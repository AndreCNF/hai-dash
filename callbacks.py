import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, State, Output
import pandas as pd
import data_utils as du
from model_interpreter.model_interpreter import ModelInterpreter
import Models
import torch
from app import app
import layouts

# [TODO] Replace with the real dataframe; this one's just a dummy one
df = pd.read_csv('data/data_n_shap_df.csv')
df = df.drop(columns='Unnamed: 0')
# Read the machine learning model
model = du.deep_learning.load_checkpoint(filepath='models/checkpoint_0.6107valloss_04_05_2020_20_25.pth', 
                                         ModelClass=Models.TLSTM)
# [TODO] Load the SHAP interpreter's expected value on the current model and dataset
expected_value = 0.5
# [TODO] Set and update the ID column name based on the chosen dataset
id_column = 'subject_id'
# [TODO] Set and update whether the model is from a custom type or not
is_custom = False
# The columns to remove from the data so that the model can use it
cols_to_remove = [0, 1]

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

# Dropdown callbacks
@app.callback(Output('dataset_name_div', 'children'),
              [Input('dataset_dropdown', 'value')])
def change_dataset(dataset):
    return dataset

@app.callback(Output('model_name_div', 'children'),
              [Input('model_dropdown', 'value')])
def change_model_name(model_name):
    return model_name

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
@app.callback(Output('salient_features_card_title', 'children'),
              [Input('instance_importance_graph', 'hoverData')])
def update_patient_salient_feat_title(hovered_data):
    # Get the selected data point's unit stay ID
    patient_unit_stay_id = hovered_data['points'][0]['y']
    return f'Patient {patient_unit_stay_id}\'s salient features'

@app.callback(Output('ts_feature_importance_card_title', 'children'),
              [Input('instance_importance_graph', 'hoverData')])
def update_ts_feat_import_title(hovered_data):
    # Get the selected data point's timestamp
    ts = hovered_data['points'][0]['x']
    return f'Feature importance on ts={ts}'

@app.callback([Output('patient_outcome_text', 'children'),
               Output('patient_outcome_card', 'color')],
              [Input('instance_importance_graph', 'hoverData')])
def update_patient_outcome(hovered_data):
    global df
    global id_column
    # Get the selected data point's unit stay ID
    patient_unit_stay_id = int(hovered_data['points'][0]['y'])
    # Filter by the selected data point
    filtered_df = df.copy()
    # [TODO] Use the right ID column according to the used dataset
    filtered_df = filtered_df[filtered_df[id_column] == patient_unit_stay_id]
    # Find if the patient dies
    patient_dies = (filtered_df.tail(1).label == 1).values[0]
    if patient_dies == True:
        outcome = 'Patient dies'
        card_color = 'danger'
    else:
        outcome = 'Patient survives'
        card_color = 'success'
    return outcome, card_color

# Plotting
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

@app.callback(Output('feature_importance_preview', 'figure'),
              [Input('dataset_name_div', 'children'),
               Input('model_name_div', 'children')])
def update_feat_import_preview(dataset_name, model_name):
    global df
    return create_feat_import_plot(df, max_display=3,
                                   xaxis_title='Average impact on output')

def create_fltd_feat_import_cards(data_filter=None):
    global df
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
              [Input('feature_importance_dropdown', 'value')])
def output_feat_import_page_cards(data_filter):
    global df
    # List that will contain all the cards that show feature importance for
    # each subset of data
    cards_list = list()
    if not isinstance(data_filter, list):
        data_filter = [data_filter]
    filters = data_filter.copy()
    if 'All' in filters:
        # Add a card with feature importance on all, unfiltered data
        cards_list.append(create_fltd_feat_import_cards()[0])
        filters.remove('All')
    if len(filters) == 0:
        # No more feature importance cards to output
        return cards_list
    # Add the remaining cards, corresponding to feature importance on possibly
    # filtered data
    [cards_list.append(card) for fltr in filters
     for card in create_fltd_feat_import_cards(fltr)]
    return cards_list

@app.callback(Output('detailed_analysis_preview', 'figure'),
              [Input('dataset_name_div', 'children'),
               Input('model_name_div', 'children')])
def update_det_analysis_preview(dataset_name, model_name):
    global df
    global model
    # Create a dataframe copy that doesn't include the feature importance columns
    column_names = [feature for feature in df.columns
                    if not feature.endswith('_shap')]
    tmp_df = df.copy()
    tmp_df = tmp_df[column_names]
    # Calculate the instance importance scores (it should be fast enough; otherwise try to do it previously and integrate on the dataframe)
    interpreter = ModelInterpreter(model, tmp_df, inst_column=1, is_custom=True)
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

@app.callback(Output('instance_importance_graph', 'figure'),
              [Input('dataset_name_div', 'children'),
               Input('model_name_div', 'children')])
def update_full_inst_import(dataset_name, model_name):
    global df
    global model
    # Create a dataframe copy that doesn't include the feature importance columns
    column_names = [feature for feature in df.columns
                    if not feature.endswith('_shap')]
    tmp_df = df.copy()
    tmp_df = tmp_df[column_names]
    # Calculate the instance importance scores (it should be fast enough; otherwise try to do it previously and integrate on the dataframe)
    interpreter = ModelInterpreter(model, tmp_df, inst_column=1, is_custom=True)
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

@app.callback(Output('salient_features_list', 'children'),
              [Input('instance_importance_graph', 'hoverData'),
               Input('dataset_name_div', 'children'),
               Input('model_name_div', 'children')])
def update_most_salient_features(hovered_data, dataset_name, model_name):
    global df
    global model
    # Get the selected data point's unit stay ID
    patient_unit_stay_id = int(hovered_data['points'][0]['y'])
    # Filter by the selected data point
    filtered_df = df.copy()
    # [TODO] Use the right ID column according to the used dataset
    filtered_df = filtered_df[filtered_df[id_column] == patient_unit_stay_id]
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
               Input('dataset_name_div', 'children'),
               Input('model_name_div', 'children')])
def update_ts_feat_import(hovered_data, dataset_name, model_name):
    global df
    global model
    global expected_value
    # [TODO] Filter by the selected data point
    # Get the selected data point's unit stay ID and timestamp
    patient_unit_stay_id = int(hovered_data['points'][0]['y'])
    ts = hovered_data['points'][0]['x']
    # Filter by the selected data point
    filtered_df = df.copy()
    # [TODO] Use the right ID column according to the used dataset
    filtered_df = filtered_df[(filtered_df[id_column] == patient_unit_stay_id)
                              & (filtered_df['ts'] == ts)]
    filtered_df = filtered_df.squeeze()
    # Get the SHAP and feature values into separate NumPy arrays
    shap_column_names = [feature for feature in df.columns
                         if feature.endswith('_shap')]
    feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
    shap_values = filtered_df[shap_column_names].to_numpy()
    features = filtered_df[feature_names].to_numpy()
    # Get the instance importance plot
    return du.visualization.shap_waterfall_plot(expected_value, shap_values, features, feature_names,
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

@app.callback(Output('final_output_graph', 'figure'),
              [Input('instance_importance_graph', 'hoverData')])
def update_final_output(hovered_data):
    global df
    global model
    global id_column
    global is_custom
    # global cols_to_remove
    # Get the selected data point's unit stay ID
    patient_unit_stay_id = int(hovered_data['points'][0]['y'])
    # Filter by the selected data point
    filtered_df = df.copy()
    # [TODO] Use the right ID column according to the used dataset
    filtered_df = filtered_df[filtered_df[id_column] == patient_unit_stay_id]
    # Only use the model-relevant features
    feature_names = [feature.split('_shap')[0] for feature in df.columns
                     if feature.endswith('_shap')]
    filtered_df = filtered_df[feature_names]
    data = torch.from_numpy(filtered_df.values)
    # Remove unwanted columns from the data
    # data = du.deep_learning.remove_tensor_column(data, cols_to_remove, inplace=True)
    # Make the data three-dimensional
    data = data.unsqueeze(0)
    # Feedforward the data through the model
    outputs = model.forward(data)
    final_output = int(float(outputs[-1]) * 100)
    # Plot the updated final output
    return du.visualization.indicator_plot(final_output, type='bullet', 
                                           higher_is_better=False,
                                           background_color=layouts.colors['gray_background'],
                                           font_color=layouts.colors['header_font_color'],
                                           font_size=20,
                                           output_type='plotly')