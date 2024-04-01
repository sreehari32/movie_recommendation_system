import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests
import streamlit.components.v1 as components


def movie_poster_fetch(movie_id):

    url = "https://api.themoviedb.org/3/movie/{}?api_key=2bb614d1e9ddaef999e871eb20076d6a".format(
        movie_id
    )
    data = requests.get(url)
    data = data.json()
    poster_path = data["poster_path"]

    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


movies_list = pickle.load(open("movie_list_df.pkl", "rb"))
similarity = pickle.load(open("simi.pkl", "rb"))
st.header("Movie Recommendation System")


def recommend(movie_name):
    recommended_movie_names_ = []
    recommended_movie_names_images_index = []
    movie_index = movies_list[movies_list["original_title"] == movie_name].index[0]
    distances = similarity[movie_index]
    movies_list_sorted = sorted(
        list(enumerate(distances)), reverse=True, key=lambda X: X[1]
    )[1:5]
    for i in movies_list_sorted:
        recommended_movie_names_.append(movies_list.iloc[i[0]]["original_title"])
        recommended_movie_names_images_index.append(
            movie_poster_fetch(movies_list.loc[i[0], "id"])
        )

    return recommended_movie_names_, recommended_movie_names_images_index


selected_movie_name = st.selectbox(
    "Please enter the movie name", movies_list["original_title"]
)
if st.button("Recommend"):
    recommended_movie_names, recommended_movie_names_images_index = recommend(
        selected_movie_name
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(recommended_movie_names[0])

        st.image(recommended_movie_names_images_index[0], use_column_width="always")

    with col2:
        st.write(recommended_movie_names[1])

        st.image(recommended_movie_names_images_index[1], use_column_width="always")

    with col3:
        st.write(recommended_movie_names[2])

        st.image(recommended_movie_names_images_index[2], use_column_width="always")

    with col4:
        st.write(recommended_movie_names[3])

        st.image(recommended_movie_names_images_index[3], use_column_width="always")
