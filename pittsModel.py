import dash
import dash_core_components as dcc 
import dash_html_components as html
from   dash.dependencies import Input,Output,State 
import random

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

colors = {
    'text': '#9F0E27'
}

# style={'background-image': 'url(/assets/bp.jpg)'},

app.layout = html.Div(style={'background-image': 'url(/assets/blur.jpg)'},children=[
    
    html.H1(children="Pitts Model",
            style={
                'textAlign':'center',
                 'color'   : colors['text'],
                 'font-weight': 'bold',
                 'font-family': 'system-ui'
            }
    ),

    html.H6(children="Enter a range",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
    dcc.Input(id='range1',value=None,type='number'),
    dcc.Input(id='range2',value=None,type='number'),

    html.H6(children="Enter the values in mentioned range",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
    
    html.Div(children=[
        
        html.H6(children="Enter Parameter 1",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p1',value=None,type='number'),

        html.H6(children="Enter Parameter 2",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p2',value=None,type='number'),

        html.H6(children="Enter Parameter 3",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p3',value=None,type='number'),

        html.H6(children="Enter Parameter 4",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p4',value=None,type='number'),

        html.H6(children="Enter Parameter 5",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p5',value=None,type='number'),

        html.H6(children="Enter Parameter 6",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p6',value=None,type='number')
        ],style={'columnCount':2}),

        html.Div(children=[
            html.Button(children="SUBMIT",id='submit-button',n_clicks=0,style={'color':'white','border':'1px solid #ddd','backgroundColor': '#1e90ff'}),
            html.Div(children="output",id='output-result',style={
            'width': '320px',
            'padding': '10px',
            'margin': '0',
            'font-family': 'system-ui',
            'font-weight': 'bold',
            'font-size':'17px'
        })
        ],style={'padding-top':'30px','padding-bottom':'70px','columnCount':2}),

        
        
])

@app.callback(
    Output('output-result','children'),
    [Input('submit-button','n_clicks')],
    [State('range1','value'),State('range2','value'),
     State('p1','value'),State('p2','value'),
     State('p3','value'),State('p4','value'),
     State('p5','value'),State('p6','value'),
     ])
def pitts_model(n_clicks,range1,range2,inp1,inp2,inp3,inp4,inp5,inp6):
    inp_args = [inp1,inp2,inp3,inp4,inp5,inp6]
    ans = ""
    
    if range1 >= range2 :
        return "former value must strictly less than latter in range\n"
    
    for val in inp_args :
        if val>range2 or val<range1 :
            return "Parameters not in Range\n"
    
    inhibitory_param1 = inp5
    inhibitory_param2 = inp6

    
    inhibitory_value = round(range1 + random.uniform(0.,1.)*(range2-range1),2)

    threshold = round(range1 + random.uniform(0.,1.)*(range2-range1),2)

    summing_func = inp1+inp2+inp3+inp4+inp5+inp6

    print(inp_args,inhibitory_value,threshold,summing_func)

    if   inhibitory_param1 < inhibitory_value :
        return "Neuron not Fired\nthreshold:{}\nSum:{}\nInhibitoryValue:{}\n".format(threshold,summing_func,inhibitory_value)
    elif inhibitory_param2 < inhibitory_value :
        return "Neuron not Fired\nthreshold:{}\nSum:{}\nInhibitoryValue:{}\n".format(threshold,summing_func,inhibitory_value)
    elif summing_func < threshold:
        return "Neuron not Fired\nthreshold:{}\nSum:{}\nInhibitoryValue:{}\n".format(threshold,summing_func,inhibitory_value)
    else :
        return "Neuron Fired!\nthreshold:{}\nSum:{}\nInhibitoryValue:{}\n".format(threshold,summing_func,inhibitory_value)

if __name__ == '__main__':
    app.run_server(debug=True)
