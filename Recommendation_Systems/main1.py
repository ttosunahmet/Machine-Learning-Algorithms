# Data Seti 1gb oldugu icin aktarma yapamadım aktarılınca kodu çalıştır.

import pandas as pd

movie = pd.read_csv("Recommendation_Systems/movie.csv")
movie = movie.loc[:,["movieId","title"]]

rating = pd.read_csv("Recommendation_Systems/rating.csv")
rating = rating.loc[:,["userId","movieId","rating"]]

data = pd.merge(movie,rating)
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")

movie_watched = pivot_table["Bad Boys (1995)"]
similarity_with_other_movies = pivot_table.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
