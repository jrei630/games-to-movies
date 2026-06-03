import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Games To Movies", page_icon="🎮", layout="centered")

st.title("🎮 Games To Movies")
st.subheader("Find movies based on the video games you love")

# Maps Steam game genres to related TMDB movie genres
GENRE_MAP = {
    'action': ['action', 'adventure', 'thriller'],
    'adventure': ['adventure', 'action', 'fantasy'],
    'rpg': ['fantasy', 'adventure', 'action'],
    'strategy': ['war', 'history', 'drama'],
    'simulation': ['drama', 'science fiction'],
    'casual': ['family', 'comedy'],
    'indie': ['drama', 'comedy'],
    'racing': ['action', 'thriller'],
    'sports': ['drama'],
    'massively multiplayer': ['action', 'science fiction'],
    'free to play': ['action', 'adventure'],
}

@st.cache_data
def load_data():
    games_raw = pd.read_csv("steam_games.csv")
    movies_raw = pd.read_csv("tmdb_5000_movies.csv")

    games = pd.DataFrame({
        'title': games_raw['name'],
        'genre': games_raw['genres'].fillna(''),
        'description': games_raw['description'].fillna('')
    })

    movies = pd.DataFrame({
        'title': movies_raw['original_title'],
        'description': movies_raw['overview'],
        'genre': movies_raw['genres'].fillna('')
    })

    games = games.dropna(subset=['title', 'description'])
    games = games[games['description'].astype(str).str.strip() != '']
    games = games.drop_duplicates(subset=['title']).reset_index(drop=True)

    movies = movies.dropna(subset=['description'])
    movies = movies[movies['description'].str.strip() != '']
    movies = movies.reset_index(drop=True)

    return games, movies

@st.cache_resource
def build_vectors(_games, _movies):
    all_text = pd.concat([_games['description'], _movies['description']])
    # ngram_range=(1,2) captures two-word phrases like "open world"
    vectorizer = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1, 2))
    vectorizer.fit(all_text)
    game_vectors = vectorizer.transform(_games['description'])
    movie_vectors = vectorizer.transform(_movies['description'])
    return game_vectors, movie_vectors

@st.cache_data
def precompute_movie_genres(_movies):
    return [set(m.lower() for m in re.findall(r'"name":\s*"([^"]+)"', str(g)))
            for g in _movies['genre']]

def game_targets(genre_str):
    out = set()
    for g in str(genre_str).lower().split(','):
        g = g.strip()
        if g:
            out.update(GENRE_MAP.get(g, [g]))
    return out

def find_game(game_title):
    query = game_title.strip().lower()
    titles = games['title'].astype(str).str.lower()
    exact = games[titles == query]
    if len(exact) > 0:
        return exact.index[0]
    partial = games[titles.str.contains(re.escape(query), na=False)]
    if len(partial) > 0:
        return partial.index[0]
    return None

def recommend(game_list, top_n=5):
    movie_scores = {}
    matched_games = []
    not_found = []

    for game_title in game_list:
        game_idx = find_game(game_title)
        if game_idx is None:
            not_found.append(game_title)
            continue

        matched_games.append(games.iloc[game_idx]['title'])
        targets = game_targets(games.iloc[game_idx]['genre'])
        game_vec = game_vectors[game_idx]
        similarities = cosine_similarity(game_vec, movie_vectors)[0]

        for i, score in enumerate(similarities):
            # genre agreement boosts proportionally — no free points
            multiplier = 1.4 if (targets & movie_genres_list[i]) else 1.0
            title = movies.iloc[i]['title']
            movie_scores[title] = movie_scores.get(title, 0) + (score * multiplier)

    ranked = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], matched_games, not_found

with st.spinner("Loading data..."):
    games, movies = load_data()
    game_vectors, movie_vectors = build_vectors(games, movies)
    movie_genres_list = precompute_movie_genres(movies)

st.markdown("---")
st.markdown("### Enter your favorite games")
st.caption("Separate multiple games with a comma")

user_input = st.text_input("", placeholder="e.g. Elden Ring, Hades, Portal 2")
num_results = st.slider("How many recommendations?", min_value=3, max_value=10, value=5)

if st.button("🎬 Get Movie Recommendations", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter at least one game.")
    else:
        game_list = [g.strip() for g in user_input.split(",")]
        with st.spinner("Finding your movies..."):
            results, matched_games, not_found = recommend(game_list, top_n=num_results)

        if matched_games:
            st.success(f"Matched games: {', '.join(matched_games)}")
        if not_found:
            st.warning(f"Could not find: {', '.join(not_found)}")

        if results:
            st.markdown("### 🎬 Recommended Movies")
            for i, (movie, score) in enumerate(results):
                st.markdown(f"**{i+1}. {movie}** — match: `{round(score, 3)}`")
        else:
            st.error("No recommendations found. Try different games.")

st.markdown("---")
st.caption("Built with Python, scikit-learn, and Streamlit | Games To Movies ML Recommender")
