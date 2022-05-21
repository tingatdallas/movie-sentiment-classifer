# movie-sentiment-classifer
LSTM based model to classify sentiment of movie reviews (IMDB).

1. Use the LSTM to build the model. Optimize the model by tuning the dropout, achieving almost 96% accuracy (check out the model at https://github.com/tingatdallas/Machine-Learning-Projects/blob/69a4ec2aefc41c0a847c44fefd7497264c0762ef/LSTM_Text%20classification_IMDB%20review%20data.ipynb).

2. The mode in this dashboard is used to evaluate the sentiment of input text or movie review. The output is probability, which is interpreted as follows :
   positive if more than 0.5, negative if less than 0,5.

3. The movie reviews are fetched from IMDB website through IMDBPY API. The dashboard layout was built with Dash / Plotly.

4. Finally, the code was dockerized and deployed to Google Cloud Run.

5. Contact me if have any question: sutingatchicago@gmail.com


