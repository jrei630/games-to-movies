import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Games To Movies", page_icon="🎮", layout="centered")

st.title("🎮 Games To Movies")
st.subheader("Find movies based on the video games you love")

# Maps game genres (RAWG) to related movie genres (TMDB)
GENRE_MAP = {
    'action': ['action', 'adventure', 'thriller'],
    'adventure': ['adventure', 'action', 'fantasy'],
    'rpg': ['fantasy', 'adventure', 'action'],
    'shooter': ['action', 'war', 'thriller'],
    'strategy': ['war', 'history', 'drama'],
    'puzzle': ['mystery', 'thriller'],
    'racing': ['action', 'thriller'],
    'sports': ['drama'],
    'simulation': ['drama', 'science fiction'],
    'fighting': ['action'],
    'arcade': ['action', 'family'],
    'platformer': ['family', 'animation', 'adventure'],
    'casual': ['family', 'comedy'],
    'indie': ['drama', 'comedy'],
    'massively multiplayer': ['action', 'science fiction'],
    'family': ['family', 'comedy', 'animation'],
    'board games': ['comedy', 'family'],
    'card': ['comedy', 'drama'],
    'educational': ['documentary', 'family'],
}

@st.cache_data
def load_data():
    games_raw = pd.read_csv("game_info_small.csv")
    movies_raw = pd.read_csv("tmdb_5000_movies.csv")

    games = pd.DataFrame({
        'title': games_raw['name'],
        'genre': games_raw['genres'].fillna(''),
        'description': (
            games_raw['genres'].fillna('') + ' ' +
            games_raw['developers'].fillna('')
        )
    })

    movies = pd.DataFrame({
        'title': movies_raw['original_title'],
        'description': movies_raw['overview'],
        'genre': movies_raw['genres'].fillna('')
    })

    games = games.dropna(subset=['title'])
    games = games[games['title'].astype(str).str.strip() != '']
    games = games.drop_duplicates(subset=['title']).reset_index(drop=True)

    movies = movies.dropna(subset=['description'])
    movies = movies[movies['description'].str.strip() != '']
    movies = movies.reset_index(drop=True)

    return games, movies

@st.cache_resource
def build_vectors(_games, _movies):
    all_text = pd.concat([_games['description'], _movies['description']])
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(all_text)
    game_vectors = vectorizer.transform(_games['description'])
    movie_vectors = vectorizer.transform(_movies['description'])
    return game_vectors, movie_vectors

def extract_movie_genres(genre_str):
    return set(m.lower() for m in re.findall(r'"name":\s*"([^"]+)"', str(genre_str)))

def extract_game_genres(genre_str):
    return [g.strip().lower() for g in str(genre_str).split('||') if g.strip()]

@st.cache_data
def precompute_movie_genres(_movies):
    return [extract_movie_genres(g) for g in _movies['genre']]

def get_target_genres(game_genres):
    """Convert a game's genres into the set of movie genres we want to match."""
    targets = set()
    for gg in game_genres:
        if gg in GENRE_MAP:
            targets.update(GENRE_MAP[gg])
        else:
            targets.add(gg)  # fall back to the raw genre name
    return targets

def recommend(game_list, top_n=5):
    movie_scores = {}
    not_found = []

    for game_title in game_list:
        game_row = games[games['title'].astype(str).str.lower() == game_title.strip().lower()]
        if len(game_row) == 0:
            not_found.append(game_title)
            continue

        game_idx = game_row.index[0]
        game_genres = extract_game_genres(games.iloc[game_idx]['genre'])
        target_genres = get_target_genres(game_genres)

        game_vec = game_vectors[game_idx]
        similarities = cosine_similarity(game_vec, movie_vectors)[0]

        for i in range(len(movies)):
            movie_genres = movie_genres_list[i]
            overlap = len(target_genres & movie_genres)
            if overlap > 0:
                # genre overlap is the main signal; similarity breaks ties
                score = overlap + similarities[i]
                title = movies.iloc[i]['title']
                movie_scores[title] = movie_scores.get(title, 0) + score

    ranked = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], not_found

# Load everything
with st.spinner("Loading data..."):
    games, movies = load_data()
    game_vectors, movie_vectors = build_vectors(games, movies)
    movie_genres_list = precompute_movie_genres(movies)

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
                st.markdown(f"**{i+1}. {movie}**")
        else:
            st.error("No recommendations found. Try different games.")

st.markdown("---")
st.caption("Built with Python, scikit-learn, and Streamlit | Games To Movies ML Recommender")
