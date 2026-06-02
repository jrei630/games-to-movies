import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Games To Movies", page_icon="🎮", layout="centered")

st.title("🎮 Games To Movies")
st.subheader("Find movies based on the video games you love")

@st.cache_data
def load_data():
    games_raw = pd.read_csv("game_info.csv")
    movies_raw = pd.read_csv("tmdb_5000_movies.csv")

    games = pd.DataFrame({
        'title': games_raw['name'],
        'genre': games_raw['genres'],
        'description': (
            games_raw['genres'].fillna('') + ' ' +
            games_raw['platforms'].fillna('') + ' ' +
            games_raw['developers'].fillna('')
        )
    })

    movies = pd.DataFrame({
        'title': movies_raw['original_title'],
        'description': movies_raw['overview'],
        'genre': movies_raw['genres']
    })

    games = games.dropna(subset=['description'])
    games = games[games['description'].str.strip() != '']
    games = games.reset_index(drop=True)

    movies = movies.dropna(subset=['description'])
    movies = movies[movies['description'].str.strip() != '']
    movies = movies.reset_index(drop=True)

    return games, movies

@st.cache_resource
def build_vectors(games, movies):
    all_descriptions = pd.concat([games['description'], movies['description']])
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(all_descriptions)
    game_vectors = vectorizer.transform(games['description'])
    movie_vectors = vectorizer.transform(movies['description'])
    return vectorizer, game_vectors, movie_vectors

with st.spinner("Loading data..."):
    games, movies = load_data()
    vectorizer, game_vectors, movie_vectors = build_vectors(games, movies)

def recommend(game_list, top_n=5):
    movie_scores = {}
    not_found = []

    for game_title in game_list:
        game_row = games[games['title'].str.lower() == game_title.strip().lower()]
        if len(game_row) == 0:
            not_found.append(game_title)
            continue
        game_idx = game_row.index[0]
        game_vec = game_vectors[game_idx]
        similarities = cosine_similarity(game_vec, movie_vectors)[0]
        for i, score in enumerate(similarities):
            title = movies.iloc[i]['title']
            movie_scores[title] = movie_scores.get(title, 0) + score

    ranked = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], not_found

st.markdown("---")
st.markdown("### Enter your favorite games")
st.caption("Separate multiple games with a comma")

user_input = st.text_input("", placeholder="e.g. Halo, Minecraft, Elden Ring")

num_results = st.slider("How many recommendations?", min_value=3, max_value=10, value=5)

if st.button("🎬 Get Movie Recommendations", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter at least one game.")
    else:
        game_list = [g.strip() for g in user_input.split(",")]
        with st.spinner("Finding your movies..."):
            results, not_found = recommend(game_list, top_n=num_results)

        if not_found:
            st.warning(f"Could not find: {', '.join(not_found)}")

        if results:
            st.markdown("### 🎬 Recommended Movies")
            for i, (movie, score) in enumerate(results):
                st.markdown(f"**{i+1}. {movie}** — score: `{round(score, 3)}`")
        else:
            st.error("No recommendations found. Try different games.")

st.markdown("---")
st.caption("Built with Python, scikit-learn, and Streamlit | Games To Movies ML Recommender")
