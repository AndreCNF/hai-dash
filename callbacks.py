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

@app.callback(Output('main-title', 'children'),
              [Input('dataset_name_div', 'children'),
               Input('model_name_div', 'children')])
def change_title(dataset_name, model_name):
    return f'{dataset_name} mortality prediction with {model_name} model'

# Plotting
@app.callback(Output('feature_importance_preview', 'figure'),
              [Input('dataset_name_div', 'children')])
def change_dataset_header(dataset):
    global df
    shap_column_names = [feature for feature in df.columns
                         if feature.endswith('_shap')]
    feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
    shap_values = df[shap_column_names].to_numpy()
    return du.visualization.shap_summary_plot(shap_values, feature_names, max_display=3,
                                              background_color=layouts.colors['gray_background'],
                                              output_type='plotly',
                                              font_family='Roboto', font_size=14,
                                              font_color=layouts.colors['body_font_color'],
                                              xaxis_title='Average impact on output')
