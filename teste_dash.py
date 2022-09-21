import os
import sys
from collections import OrderedDict
sys.path.insert(0, r'C:\Users\lucas.oliveira\Documents\enterpython')
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import pandas_datareader.data as web
import datetime
import pandas as pd
from datetime import datetime
from graphs_interface import ploty_basic_API
from class_alocacao import LowVol, generae_statistcs_interface
import site

print("aloha")
print(os.path.dirname(sys.executable))
print(site.getsitepackages()[0])


data = OrderedDict(
    [
        ("Date", ["2015-01-01", "2015-10-24", "2016-05-10", "2017-01-10", "2018-05-10", "2018-08-15"]),
        ("Region", ["Montreal", "Toronto", "New York City", "Miami", "San Francisco", "London"]),
        ("Temperature", [1, -20, 3.512, 4, 10423, -441.2])

    ]
)

df = pd.DataFrame(data)

class teste(object):

    def __init__(self):
        self.parametro = 1
        self.df2 = self.pp()

    def pp(self):
        d1 = datetime(2025, 1, 1).date()
        dts = [d1] * 6
        tickers = ['AMZN', 'GOOGL', 'FB', 'PFE', 'MRNA', 'BNTX']
        cc = [1] * 6
        df = pd.DataFrame(
            {"Date": dts, 'Symbols': tickers, 'Close': cc, 'High': cc, 'Low': cc, 'Open': cc, 'Volume': cc})
        print("ola ola")
        return df


# https://stooq.com/
start = datetime(2020, 1, 1)
end = datetime(2020, 12, 3)
df = web.DataReader(['AMZN', 'GOOGL', 'FB', 'PFE', 'MRNA', 'BNTX'],
                    'stooq', start=start, end=end)
# df=df.melt(ignore_index=False, value_name="price").reset_index()
# df = df.stack().reset_index()
# print("before")
# print(df[:15])
# df.columns = ['Date','Symbols','Close','High','Low','Open','Volume']
# print(df[:15])

d1 = datetime(2020, 1, 1).date()
dts = [d1] * 6
tickers = ['AMZN', 'GOOGL', 'FB', 'PFE', 'MRNA', 'BNTX']
cc = [1] * 6
df = pd.DataFrame({"Date": dts, 'Symbols': tickers, 'Close': cc, 'High': cc, 'Low': cc, 'Open': cc, 'Volume': cc})

# df.to_csv("mystocks.csv", index=False)
# df = pd.read_csv("mystocks.csv")
# print(df[:15])


# https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

aa = teste()

lv = LowVol(index='IBX', country='brazil', type_trades=["long", "short"], flag_signal=False, local_database=True,
            dict_param='gss', nbin=7, backtest_di=True)
df_params_pure, df_params_full_pure, tam_rank_in, tam_rank_out = lv.get_data()
lv.df_params_full_pure = df_params_full_pure.copy()

# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
app.layout = dbc.Container([

    dbc.Row(
        dbc.Col(html.H1("Portfolio Simulator",
                        className='text-center text-primary mb-4'),
                width=12)
    ),

    dbc.Row([

        dbc.Col([

            html.Label(['Weight settings:'], style={'font-weight': 'bold', "text-align": "left", "width": "45%"}),
            html.Div(dcc.Slider(id="s1", min=0, value=0.3, max=1,
                                marks={0: {'label': 'IMAB5+', 'style': {'color': '#77b0b1'}}},
                                tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(dcc.Slider(id="s2", min=0, value=0.2, max=1,
                                marks={0: {'label': 'IMAB5', 'style': {'color': '#77b0b1'}}},
                                tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(dcc.Slider(id="s3", min=0, value=0.15, max=1,
                                marks={0: {'label': 'DI_PRE_3Y', 'style': {'color': '#77b0b1'}}},
                                tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(dcc.Slider(id="s4", min=0, value=0.05, max=1,
                                marks={0: {'label': 'CDI', 'style': {'color': '#77b0b1'}}},
                                tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(
                dcc.Slider(id="s5", min=0, value=0.3, max=1, marks={0: {'label': 'RV', 'style': {'color': '#77b0b1'}}},
                           tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(
                dcc.Slider(id="s6", min=0, value=0.7, max=1, marks={0: {'label': 'IBX', 'style': {'color': '#77b0b1'}}},
                           tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(dcc.Slider(id="s7", min=0, value=0.1, max=1,
                                marks={0: {'label': 'SMALL11', 'style': {'color': '#77b0b1'}}},
                                tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(dcc.Slider(id="s8", min=0, value=0.1, max=1,
                                marks={0: {'label': 'DIV011', 'style': {'color': '#77b0b1'}}},
                                tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(dcc.Slider(id="s9", min=0, value=0.1, max=1,
                                marks={0: {'label': 'S&P500', 'style': {'color': '#77b0b1'}}},
                                tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            html.Div(dcc.Slider(id="s10", min=0, value=0.0, max=1,
                                marks={0: {'label': 'Fund', 'style': {'color': '#77b0b1'}}},
                                tooltip={"placement": "bottom", "always_visible": True}), style={"width": "45%"}),
            dcc.Graph(id='line-fig', figure={})

        ],  # width={'size':5, 'offset':1, 'order':1},

            #             html.Div(
            #             children=[
            #                 html.P('Description:',
            #                     style={
            #                         'fontSize': 17
            #                     }
            #                 ),
            #                 html.P(id='output_text',
            #                     style={
            #                         'fontSize': 40
            #                     }
            #                 )
            #             ],
            #             style={
            #                 'width': '25%',
            #                 'display': 'inline-block'
            #             }
            #             ),

            xs=12, sm=12, md=12, lg=5, xl=5
        ),
        # dbc.Col([
        #             html.Div(
        #             children=[
        #                 html.P('Description:',
        #                     style={
        #                         'fontSize': 17
        #                     }
        #                 ),
        #                 html.P(id='output_text',
        #                     style={
        #                         'fontSize': 40
        #                     }
        #                 )
        #             ],
        #             style={
        #                 'width': '25%',
        #                 'display': 'inline-block'
        #             }
        #             ),

        #     ]),
        dbc.Col([

            html.Label(['Date Range:'], style={'font-weight': 'bold', "text-align": "left", "width": "45%"}),
            html.Button('Run Backtest', id='btn-nclicks-1', n_clicks=0),
            html.Div(id='intermediate-value', style={'display': 'none'}),
            html.Div(
                dcc.DatePickerRange(
                    id='date_range_ob',
                    min_date_allowed=datetime(2013, 1, 1).date(),
                    max_date_allowed=datetime(2022, 9, 11).date(),
                    initial_visible_month=datetime(2013, 1, 1).date(),
                    start_date=datetime(2013, 1, 1).date(),
                    end_date=datetime(2022, 9, 11).date()
                ),
                style={"width": "50%"}),

            html.Label(['Rebalance Freq:'], style={'font-weight': 'bold', "text-align": "left", "width": "45%"}),
            html.Br(),
            html.Div(
                dcc.Dropdown(id='my-dpdnzz', multi=False, value='1Y',
                             options=[{'label': x, 'value': x}
                                      for x in ['6M', '1Y', '2Y']],
                             ), style={"width": "25%"}),

            html.Label(['# Iterations:'], style={'font-weight': 'bold', "text-align": "left", "width": "45%"}),
            html.Br(),
            html.Div(
                dcc.Dropdown(id='my-dpdn2', multi=False, value=0,
                             options=[{'label': x, 'value': x}
                                      for x in [0, 1, 2, 3, 4, 5]],
                             ), style={"width": "25%"}),

            ########
            html.Label(['Simulation Variable:'], style={'font-weight': 'bold', "text-align": "left", "width": "45%"}),
            html.Br(),
            html.Div(
                dcc.Dropdown(id='sim_var', multi=False, value='Fund',
                             options=[{'label': x, 'value': x}
                                      for x in ['Fund', 'IMAB5', 'IMAB5_P', 'DI_PRE_3Y', 'CDI', 'RV', 'S&P', 'SMALL11',
                                                'DIVO11']],
                             ), style={"width": "25%"}),

            ###
            # html.Br(),
            # html.Br(),
            html.Br(),
            # html.Br(),
            # html.Br(),
            # html.Br(),
            # html.Div(

            #         #"text-align": "right",
            #         children=[
            #             html.P('Sum of Weights:',
            #                 style={
            #                     'fontSize': 17
            #                 }
            #             ),
            #             html.P(id='output_text',
            #                 style={
            #                     'fontSize': 40
            #                 }
            #             )
            #         ],
            #         style={
            #             'width': '25%',
            #             'display': 'inline-block'
            #         }
            #         ),

            dbc.Col([

                html.Div(
                    children=[
                        html.P('Sum of Weights:',
                               style={
                                   'fontSize': 17
                               }
                               ),
                        html.P(id='output_text',
                               style={
                                   'fontSize': 40
                               }
                               )
                    ],
                    style={
                        'width': '25%',
                        'display': 'inline-block'
                    }
                ),

                html.Div(
                    children=[
                        html.P('Sum of Weights RV:',
                               style={
                                   'fontSize': 17
                               }
                               ),
                        html.P(id='output_text2',
                               style={
                                   'fontSize': 40
                               }
                               )
                    ],
                    style={
                        'width': '25%',
                        'display': 'inline-block'
                    }
                )

            ]),

            dcc.Graph(id='line-fig2', figure={}, )
        ],  # width={'size':5, 'offset':0, 'order':2},
            xs=12, sm=12, md=12, lg=5, xl=5, style={"width": "45%"}
        ),

    ], justify='start'),  # Horizontal:start,center,end,between,around

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line-fig3', figure={})
        ],  # width={'size':5, 'offset':1},
            xs=12, sm=12, md=12, lg=5, xl=5, style={"width": "45%"}
        ),

        dbc.Col([

            dash_table.DataTable(
                # data=generae_statistcs_interface(pd.read_json(jsonified_cleaned_data)).to_dict('records'),
                id='tabelao',
                # data = df.to_dict('records'),
                data=[],
                # columns=[{'id': c, 'name': c} for c in df.columns],
                columns=[],
                style_data_conditional=[
                    {
                        'if': {
                            'column_id': 'Avg_Improvement`',
                            'filter_query': ' {Avg_Improvement} > 0'
                        },
                        'backgroundColor': '#3D9970',
                        'color': 'green',
                    },
                ],
                style_as_list_view=True,
                style_cell={'padding': '5px'},  # style_cell refers to the whole table
                style_header={
                    'backgroundColor': 'gray',
                    'fontWeight': 'bold',
                    'border': '1px solid black'
                }, style_data={'border': '1px solid gray'})

        ],  # width={'size':5, 'offset':1},
            xs=12, sm=12, md=12, lg=5, xl=5
        )
    ], align="center")  # Vertical: start, center, end

], fluid=True)


# variable_or_field_to_show
@app.callback(
    Output('output_text2', 'children'),
    Input('s6', 'value'),
    Input('s7', 'value'),
    Input('s8', 'value'),
    Input('s9', 'value'),
)
def clean_data2(s6, s7, s8, s9):
    return s6 + s7 + s8 + s9


# variable_or_field_to_show
@app.callback(
    Output('output_text', 'children'),
    Input('s1', 'value'),
    Input('s2', 'value'),
    Input('s3', 'value'),
    Input('s4', 'value'),
    Input('s5', 'value'),
    Input('s10', 'value'),
)
def clean_data2(s1, s2, s3, s4, s5, s10):
    return s1 + s2 + s3 + s4 + s5 + s10


################################# UPDATING NONE DIV ##########################
@app.callback(
    Output('intermediate-value', 'children'),
    Input('btn-nclicks-1', 'n_clicks'),
    State('my-dpdn2', 'value'),
    State('sim_var', 'value'),
    State('s1', 'value'),
    State('s2', 'value'),
    State('s3', 'value'),
    State('s4', 'value'),
    State('s5', 'value'),
    State('s6', 'value'),
    State('s7', 'value'),
    State('s8', 'value'),
    State('s9', 'value'),
    State('s10', 'value'),
    State('date_range_ob', 'start_date'),
    State('date_range_ob', 'end_date')
)
def clean_data(if_click, n_inter, sim_var, imab5p_w, imab5_w, dipre_w, cdi_w, rv_w, ibx_w, small11_w, divo11_w, sp500_w,
               fund_w, start_date, end_date):
    df = lv.grafico_iteracoes_fundo(lv.df_params_full_pure,
                                    lista_pesos_fundo=[0, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25][0:(n_inter + 1)],
                                    if_return_df=True, ativo_variavel=sim_var, dt_min=start_date, dt_max=end_date,
                                    imab5p_w=imab5p_w, imab5_w=imab5_w, dipre_w=dipre_w, cdi_w=cdi_w, rv_w=rv_w,
                                    ibx_w=ibx_w, small11_w=small11_w, divo11_w=divo11_w, sp500_w=sp500_w, fund_w=fund_w)

    return df.to_json()


################################# UPDATING NONE DIV ##########################


################################### GRAFICO 1 ###########################

@app.callback(
    Output('line-fig', 'figure'),
    Input('intermediate-value', 'children'))
def update_graph(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data)
    figln2 = ploty_basic_API(dff.drop("cota_cdi", axis=1), "date", dff.drop("cota_cdi", axis=1).columns.tolist()[1:],
                             x_title='Date', y_title='Pl', title="Fund's PL over time")
    return figln2


################################### GRAFICO 1 ###########################


################################### GRAFICO 22222222222222 ###########################
@app.callback(
    Output('line-fig2', 'figure'),
    Input('intermediate-value', 'children'))
def update_graph(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data)
    dff = generae_statistcs_interface(dff.drop("cota_cdi", axis=1), dff[["date", "cota_cdi"]], ret_dd=True)
    figln2 = ploty_basic_API(dff, "date", dff.columns.tolist()[1:], x_title='Date', y_title='MaxDD(%)',
                             title="Fund's MaxDrawdown over time")
    return figln2


################################### GRAFICO 22222222222222 ###########################


################################### GRAFICO 333333333 ###########################
@app.callback(
    Output('line-fig3', 'figure'),
    Input('intermediate-value', 'children'))
def update_graph(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data)
    figln2 = ploty_basic_API(dff.drop("cota_cdi", axis=1), "date", dff.drop("cota_cdi", axis=1).columns.tolist()[1:])
    return figln2


################################### GRAFICO 333333333 #################


################################### TABELAO ###########################
@app.callback(
    Output('tabelao', 'data'),
    Output('tabelao', 'columns'),
    Input('intermediate-value', 'children'))
def update_graph(jsonified_cleaned_data):
    # df row
    dff = pd.read_json(jsonified_cleaned_data)

    # df processed
    df = generae_statistcs_interface(dff.drop("cota_cdi", axis=1), dff[["date", "cota_cdi"]])
    data = df.to_dict('records')
    columns = [{'id': c, 'name': c} for c in df.columns]
    return data, columns


################################### TABELAO ###########################


if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
