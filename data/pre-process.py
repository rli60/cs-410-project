import pandas as pd

#load movies.csv into a data frame
movie_df = pd.read_csv("ml-latest/movies.csv")

#extract date into a column and remove date from title
movie_df['year'] = movie_df['title'].str.extract("\((\d{4})\)$", expand=False)

movie_df['title'] = movie_df['title'].str.replace("\((\d{4})\)$", '', regex=True).str.strip()

#load tags.csv into a data frame
tag_df = pd.read_csv("ml-latest/tags.csv")

#drop rows with missing tags
tag_df = tag_df.dropna(subset='tag')

#only keep one copy of each tag per movie (remove duplicates)
tag_df = tag_df.drop_duplicates(subset=['tag', 'movieId'])

#group each tag by movie id and then join each of the tags for a movie
tags = tag_df.groupby('movieId')['tag'].aggregate(' '.join).reset_index()

#load ratings.csv into a data frame
rating_df = pd.read_csv("ml-latest/ratings.csv")

#drop rows with missing ratings
rating_df = rating_df.dropna(subset='rating')

#group each rating by movie id and compute rating count and average rating for each movie
ratings = rating_df.groupby('movieId')['rating'].aggregate(['count', 'mean']).reset_index()

#rename columns
ratings = ratings.rename(columns={'count':'rating_count', 'mean':'average_rating'})

#load links.csv into a data frame
link_df = pd.read_csv("ml-latest/links.csv")

#merge the tags, ratings, and links with their matching movieIds
movie_df = movie_df.merge(right=tags, how='left', on='movieId') 
movie_df = movie_df.merge(right=ratings, how='left', on='movieId') 
movie_df = movie_df.merge(right=link_df, how='left', on='movieId') 

#fill in any empty tags or ratings counts
movie_df['tag'] = movie_df['tag'].fillna('')
movie_df['rating_count'] = movie_df['rating_count'].fillna(0)

print(movie_df.shape)
movie_df.to_csv('movies_cleaned.csv', index=False)