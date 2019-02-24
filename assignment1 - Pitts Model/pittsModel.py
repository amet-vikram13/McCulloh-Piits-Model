import dash
import dash_core_components as dcc
import dash_html_components as html
from   dash.dependencies import Input,Output,State
import random
import base64

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

colors = {
    'text': '#9F0E27'
}

# style={'background-image': 'url(/assets/bp.jpg)'},

image_filename = './assets/pikachu.jpg' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div(style={'background-image': 'url(/assets/blur.jpg)'},children=[

    html.H1(children="Pokemon Wars",
            style={
                'textAlign':'center',
                 'color'   : colors['text'],
                 'font-weight': 'bold',
                 'font-family': 'system-ui'
            }
    ),

    html.H6(children="Enter a range for all the input fields",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
    dcc.Input(id='range1',value=None,type='number'),
    dcc.Input(id='range2',value=None,type='number'),

    html.H6(children="How much do you think is your Pikachu's power? (Within above specified limits)\nThe opponent is Squirtle",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),

    html.Div(children=[

        html.H6(children="Attack",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p1',value=None,type='number'),

        html.H6(children="Defence",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p2',value=None,type='number'),

        html.H6(children="Power",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p3',value=None,type='number'),

        html.H6(children="Moves",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p4',value=None,type='number'),

        html.H6(children="Special Power",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p5',value=None,type='number'),

        html.H6(children="Strength",style={'color':'#9F0E9F','font-family': 'system-ui','font-weight': 'bold'}),
        dcc.Input(id='p6',value=None,type='number')
        ],style={'columnCount':2}),

        html.Div(children=[
            html.Button(children="Play",id='submit-button',n_clicks=0,style={'color':'white','border':'1px solid #ddd','backgroundColor': '#1e90ff'})

        ],style={'padding-top':'30px','padding-bottom':'20px'}),

        html.Div(children=[
            html.Div(children="output",id='output-result',style={
            'width': '320px',
            'padding': '10px',
            'margin': '0',
            'font-family': 'system-ui',
            'font-weight': 'bold',
            'font-size':'17px',
            'border': 'solid 3px',
            'margin-top': '20px'
        }),
            html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode()), style={'width': '500px'})
        ],style={'padding-top':'10px','padding-bottom':'70px', 'columnCount': 2})

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

    #inhibitory_value = (range1 + range2)/2
    #threshold = (range1 + range2)/2

    val1 = -1
    val2 = -1
    val3 = -1
    val4 = -1
    val5 = -1
    val6 = -1

    if inp1 < threshold:
        val1 = 0
    else:
        val1 = 1

    if inp2 < threshold:
        val2 = 0
    else:
        val2 = 1

    if inp3 < threshold:
        val3 = 0
    else:
        val3 = 1

    if inp4 < threshold:
        val4 = 0
    else:
        val4 = 1

    #inp5 and 6 are inhibitory
    if inp5 < threshold:
        val5 = 1
    else:
        val5 = 0

    if inp6 < threshold:
        val6 = 1
    else:
        val6 = 0

    sum = val1+val2+val3+val4

    #print(inp_args,inhibitory_value,threshold,summing_func)

    if  val5 == 1:
        return "Your Pikachu will likely lose the battle\n\nthreshold:{}\n\nSum:{}\nInhibitoryValue:{}\n\n".format(threshold,sum,inhibitory_value)
    elif val6 == 1:
        return "Your Pikachu will likely lose the battle\n\nthreshold:{}\n\nSum:{}\nInhibitoryValue:{}\n\n".format(threshold,sum,inhibitory_value)
    elif sum < 3:
        return "Your Pikachu will likely lose the battle\n\nthreshold:{}\n\nSum:{}\nInhibitoryValue:{}\n\n".format(threshold,sum,inhibitory_value)
    else:
        return "Your Pikachu will likely win the battle\n\nthreshold:{}\n\nSum:{}\nInhibitoryValue:{}\n\n".format(threshold,sum,inhibitory_value)

if __name__ == '__main__':
    app.run_server(debug=True)
