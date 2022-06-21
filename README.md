# movie-sentiment-classifer
LSTM based model to classify sentiment of movie reviews (IMDB).

1. Use the LSTM to build the model. Optimize the model by tuning the dropout, achieving almost 96% accuracy.

2. The mode in this dashboard is used to evaluate the sentiment of input text or movie review. The output is probability, which is interpreted as follows :
   positive if more than 0.5, negative if less than 0.5 .

3. The movie reviews are fetched from IMDB website through IMDBPY API. The dashboard layout was built with Dash / Plotly.

4. Finally, the code was dockerized and deployed to Google Cloud Run (https://sutingatchicago-mypbgqf64a-uc.a.run.app/).

5. The web app is easily adapted to other sentiment analysis such as reviews in twitter, products reviews in E-commence, etc.

6. Contact me if have any question: sutingatchicago@gmail.com


