import gc
from dash import Dash, html, dcc,Input, Output, callback #dash == 2.3.1
import dash_bootstrap_components as dbc # dash_bootstrap_components==1.1.0
import dash
import plotly.express as px

from imdb import IMDb
import tensorflow as tf
from statistics import mean
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# gunicorn==20.1.0

# https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
#server = app.server

# Define layouts. Import LSTM model. Define functions to fetch, clean and process movie reviews data, which feed into model for sentiment analysis.
# Load the IMDB review classification model
model=tf.keras.models.load_model("sentiment-model")

# Define functions
# Define function to fetch review by movie name

ia = IMDb()
def get_review(movie_name):
    movies=ia.search_movie(movie_name)
    theMatrix = ia.get_movie(movies[0].movieID)
    ia.update(theMatrix,['reviews'])
    try:
        text=theMatrix['reviews']
        text2=[]
        for i in text:
            text2.append(i['content'])
        return text2
    except:
        return None

# Define function to get all the movies performed by a actor
def get_person_movies(person_name):
    person=ia.search_person(person_name)
    #person_id=person[0].personID
    char=ia.get_person_biography(person[0].personID)
    movies=[*char['titlesRefs']]

    return movies

# Define function to analyze review sentiment (rating)
def get_movie_rating(movie_review):
    #model=tf.keras.models.load_model("sentiment-model")
    prediction=model.predict(movie_review)
    #del model
    #gc.collect()
    prediction=[list(x)[0] for x in prediction]
    review_rating_dict=dict(zip(movie_review,prediction))

    return mean(prediction), review_rating_dict

# Define function to get sentiment of all movies sentiment by an actor.
def get_rating_person(person_name):
    movie_list=get_person_movies(person_name)
    movie_rating={}
    for movie in movie_list:
        reviews=get_review(movie)
        if reviews!=None:
            i=get_movie_rating(reviews)
            movie_rating[movie]=i
    return movie_rating


# 1. layout1_text
# style and text in this layout
style_input1={'width':'25rem','height':'15rem','border':'1px solid','font-size':'1.1rem','color':'lightblue',
             'border-radius':'10px','border-color':'green','background':'#F5F5F5','text-align':'left'}

text1="""This app illustrates powerfulness of the artificial neural network (specifically deep learning) in text classification.
The algorithm used here is called long short-term memory (LSTM), which is proven to be very effective in dealing with the sequence data like text, time-series 
data. A LSTM-based model was built here to predict sentiment of movie reviews. Through carefully fine-tuning the model, a 99% accuracy is achieved in test IMDB dataset."""
text2="""In the upper left panel, the LSTM algorithm is able to predict the sentiment of input text in real-time manner. TRY IT OUT."""
text3="""In the upper right panel, you can enter a movie name. The app will automatically fetch movie reviews from IMDB, and then feed into the LSTM model for rating in term of their sentiment."""
text4="""In the lower left panel, you can enter an actor's name. The app will automatically retrieve reviews from all the movies for this actor,
 and then feed into the LSTM model for rating in term of their sentiment. Note: it may takes few minutes"""
text5="""The lower right panel ranks most famous Holyhood actors based on their movie reviews assessed by the LSTM model."""
text6="""IMDB only allows fetching of 25 reviews for a movie. Therefore, this 25 reviews may not be representative of a movie in some cases."""

layout1_text=dbc.Col([
    html.H5("Real-time sentiment analysis",
            className='text-center md-4'),

    html.Div(
        [
            dcc.Textarea(id='text',
                         placeholder="Enter some a few thing here to see how the algorithm responds, like this movie is amazing, or it is a bad movie.",
                         value='it is an amazing movie',className='text-left text-primary md-4'
                         ,style=style_input1),
        ],style={'display': 'flex','justify-content':'center','gap':'1rem','margin-bottom':'1rem'},
    ),

    html.Div(id='text_rating'),

    html.Div([
        html.H4("Notes",
                className='text-center md-4'),
        html.P(text1,style={'font-size':'1.2rem','text-align':'left'}),
        html.P(text2,style={'font-size':'1.2rem','text-align':'left'}),
        html.P(text3,style={'font-size':'1.2rem','text-align':'left'}),
        html.P(text4,style={'font-size':'1.2rem','text-align':'left'}),
        html.P(text5,style={'font-size':'1.2rem','text-align':'left'}),
        html.P(text6,style={'font-size':'1.2rem','text-align':'left','color':'red'}),
        html.P('Copyright © 2022 Ting Su. All Rights Reserved. contact: sutingatchicago@gmail.com')
    ])
],xs=12, sm=12, md=12, lg=6, xl=6,style={'border':'1px solid black','padding-top':'0.5rem'}
)


# 2. layout2_movie_name

# Style in this layout
style_input2={'width':'25rem','height':'2rem','border':'1px solid','color':'lightblue',
             'border-radius':'10px','border-color':'green','background':'#F5F5F5'}

layout2_movie_name=dbc.Col([
    html.H5("Check movie sentiment",
            className='text-center md-4'),

    html.Div([
        dbc.Label('Enter movie name:',className='text-center mb-4 bold'),
        dcc.Input(id='movie_name2',
                  placeholder='Enter movie like Forest gump, Radhe',
                  value='',className='text-center text-primary md-4'
                  ,style=style_input2),
        html.Div(id='overall_rating')
    ],style={'display': 'flex','justify-content':'center','gap':'1rem'},
    ),
    html.Div(id='movie_review'),
],xs=12, sm=12, md=12, lg=6, xl=6,style={'border':'1px solid black','padding-top':'0.5rem'}
)

# 3. layout3_movie_actor
# style in this layout
style_input3={'width':'25rem','height':'2rem','border':'1px solid','color':'lightblue',
             'border-radius':'10px','border-color':'green','background':'#F5F5F5'}

# Layout
layout3_movie_actor=dbc.Col([
    html.H5("Check movies' sentiment by an actor",className='text-center md-4'),

    html.Div([
        dbc.Label('Enter actor name:',className='text-center mb-4'),
        dcc.Input(id='actor_name',
                  placeholder='Enter actor like Jennifer Lawrence, Tom Hanks or Bradley Cooper',
                  value='',className='text-center text-primary md-4'
                  ,style=style_input3),

    ],style={'display': 'flex','justify-content':'center','gap':'1rem'},
    ),

    html.Div(id='review-classification'),
],xs=12, sm=12, md=12, lg=6, xl=6,style={'border':'1px solid black','padding-top':'0.5rem'}
)


# 4. layout3_rank_actors
# import processed ranking from excel file and plot the figure
excel=pd.read_excel("new.xlsx")
excel.dropna(axis=1,inplace=True)
excel1=excel[['Name','Rating']].sort_values('Rating')
excel1.reset_index(inplace=True)
excel1.drop(columns=['index'],inplace=True)

fig4=px.bar(excel1,y=excel1.Rating,x=excel1.Name,color='Rating')
fig4.update_layout(
    height=800,
    font=dict(
        family="sans-serif",
        size=15,
        color="black"),)

fig4.update_xaxes(title_text="Name", title_font=dict(size=25),tickfont=dict(size=10))
fig4.update_yaxes(title_text="Rating", title_font=dict(size=25),tickfont=dict(size=20),range=[0.3,1])

# Layout
layout4_rank_actors=dbc.Col([
    html.H5("Rank of stars by movie rating",className='text-center md-4'),
    dcc.Graph(figure=fig4)
],xs=12, sm=12, md=12, lg=6, xl=6,style={'border':'1px solid black','padding-top':'0.5rem'})


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
app.layout = html.Div([

    dbc.Row(
        dbc.Col(html.H1("Sentiment analysis of movie reviews using LSTM-based model",
                        className='text-center mb-4',style={'color':'LightSeaGreen'}),
                width=12)
    ),

    dbc.Row([
        layout1_text,
        layout2_movie_name,
        ]),
    dbc.Row([
        layout3_movie_actor,
        layout4_rank_actors,
    ]),
    
],style={'margin-top':'2rem','margin-left':'3rem','margin-right':'3rem'})


# layout1 callback
# callback function to return sentiment (or rating) of the input text.
@callback(
    Output('text_rating', 'children'),
    Input('text', 'value'),
)

def update_graph(input1): # Plot candlestick price
    input1=[input1]
    rating,rating_dict=get_movie_rating(input1)
    #rating=round(rating,5)
    df=pd.DataFrame.from_dict(rating_dict, orient='index',columns=['Positive Probability (Sentiment 0 to 1)'])
    #df['review']=df.index
    df.reset_index(drop=True,inplace=True)
    df['Class']=df['Positive Probability (Sentiment 0 to 1)'].map(lambda x:'positive' if x> 0.5 else 'negative')
    print('test')
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True,color="info")


# layout2. Callback function to return movie review sentiment.

@callback(
    Output('movie_review', 'children'),
    Output('overall_rating','children'),
    Input('movie_name2', 'value'),
)

def update_graph2(input1): # Plot candlestick price
    #print('test1')
    rating,rating_dict=get_movie_rating(get_review(input1))
    rating=round(rating,5)
    #print(rating)
    df=pd.DataFrame.from_dict(rating_dict, orient='index',columns=['Positive Probability (Sentiment 0 to 1)'])
    df['review']=df.index
    df.reset_index(drop=True,inplace=True)
    df['Class']=df['Positive Probability (Sentiment 0 to 1)'].map(lambda x:'positive' if x> 0.5 else 'negative')
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True,color="info"), [html.H5("Average rating:", style={'color': 'red','text-align':'center'}),html.H5(rating, style={'color': 'red'})]

# lay3.  Callback function to get sentiment of movies by an actor based on the reviews.

@callback(
    Output('review-classification', 'children'),
    Input('actor_name', 'value'),
)

def update_graph3(input1): # Plot candlestick price
    movie_rating_dict=get_rating_person(input1)
    df=pd.DataFrame.from_dict(movie_rating_dict, orient='index',columns=['Rating'])
    df['Movies']=df.index
    df.reset_index(drop=True,inplace=True)
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True,color="info")



if __name__ == "__main__":
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')


    #app.run_server(debug=True, port=8080)
