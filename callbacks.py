import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, State, Output
import pandas as pd
import data_utils as du
from app import app
import layouts

# [TODO] Replace with the real dataframe; this one's just a dummy one
df = pd.read_csv('data/data_n_shap_df.csv')

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

# Plotting
def create_feat_import_plot(df, max_display=None,
                            xaxis_title='mean(|SHAP value|) (average impact on model output magnitude)'):
    shap_column_names = [feature for feature in df.columns
                         if feature.endswith('_shap')]
    feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
    shap_values = df[shap_column_names].to_numpy()
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
