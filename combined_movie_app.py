import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Movie Recommendations System — no API token or internet needed
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

LANGUAGES = {
    "English": {"code": "en", "aliases": {"en", "english"}},
    "Telugu": {"code": "te", "aliases": {"te", "telugu"}},
    "Hindi": {"code": "hi", "aliases": {"hi", "hindi"}},
    "Tamil": {"code": "ta", "aliases": {"ta", "tamil"}},
}

st.set_page_config(page_title="Movie Recommendations", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendations System")
st.caption("No TMDB API • Works with local CSV files • Recommendations are filtered by language first")

@st.cache_data
def load_data():
    """Load optional user catalogues and the MovieLens small dataset if present."""
    frames = []
    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.glob("*.csv")):
            try:
                df = pd.read_csv(path)
                # A combined catalogue can use the documented columns below.
                if "title" in {str(c).strip().lower() for c in df.columns}:
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    frames.append(df)
            except Exception:
                pass

    # Support the common MovieLens ml-latest-small folder structure.
    for movies_path, ratings_path in [
        (APP_DIR / "ml-latest-small" / "movies.csv", APP_DIR / "ml-latest-small" / "ratings.csv"),
        (APP_DIR / "movies.csv", APP_DIR / "ratings.csv"),
    ]:
        if movies_path.exists():
            try:
                movies = pd.read_csv(movies_path)
                movies.columns = [str(c).strip().lower() for c in movies.columns]
                if "title" in movies.columns:
                    if ratings_path.exists():
                        ratings = pd.read_csv(ratings_path)
                        ratings.columns = [str(c).strip().lower() for c in ratings.columns]
                        if {"movieid", "rating"}.issubset(ratings.columns) and "movieid" in movies.columns:
                            summary = ratings.groupby("movieid")["rating"].agg(["mean", "count"]).reset_index()
                            summary.columns = ["movieid", "rating", "rating_count"]
                            movies = movies.merge(summary, on="movieid", how="left")
                    frames.append(movies)
            except Exception:
                pass

    if not frames:
        return pd.DataFrame(columns=["title", "language", "genres", "rating", "rating_count", "year"])

    normalized = []
    for df in frames:
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "title" not in df.columns:
            continue
        if "language" not in df.columns:
            # MovieLens has no reliable language field. Do not guess from title.
            df["language"] = ""
        if "genres" not in df.columns:
            df["genres"] = ""
        if "rating" not in df.columns:
            for candidate in ("average_rating", "vote_average", "mean_rating", "mean"):
                if candidate in df.columns:
                    df["rating"] = df[candidate]
                    break
        if "rating_count" not in df.columns:
            for candidate in ("ratings_count", "vote_count", "count"):
                if candidate in df.columns:
                    df["rating_count"] = df[candidate]
                    break
        if "year" not in df.columns:
            df["year"] = ""
        if "overview" not in df.columns:
            df["overview"] = ""
        if "director" not in df.columns:
            df["director"] = ""
        for col in ("rating", "rating_count"):
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("title", "language", "genres", "year", "overview", "director"):
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)
        normalized.append(df[["title", "language", "genres", "rating", "rating_count", "year", "overview", "director"]])

    if not normalized:
        return pd.DataFrame(columns=["title", "language", "genres", "rating", "rating_count", "year", "overview", "director"])

    all_movies = pd.concat(normalized, ignore_index=True)
    all_movies["language_key"] = all_movies["language"].str.strip().str.lower()
    all_movies["title_key"] = all_movies["title"].str.strip().str.casefold()
    # Keep one entry per title/language, preferring the row with a rating.
    all_movies["_has_rating"] = all_movies["rating"].notna().astype(int)
    all_movies = all_movies.sort_values("_has_rating", ascending=False)
    all_movies = all_movies.drop_duplicates(["title_key", "language_key"], keep="first")
    return all_movies.drop(columns=["_has_rating"]).reset_index(drop=True)

movies = load_data()

def language_matches(value, language):
    value = str(value).strip().lower()
    return value in LANGUAGES[language]["aliases"]

def rating_label(row):
    rating = row.get("rating")
    count = row.get("rating_count")
    if pd.notna(rating):
        label = f"{float(rating):.1f}/10"
        if pd.notna(count):
            label += f" ({int(count):,} ratings)"
        return label
    return "Not available in local data"

def display_movie(row):
    st.markdown(f"### {row['title']}")
    bits = []
    if str(row.get("year", "")).strip():
        bits.append(str(row["year"]))
    if str(row.get("genres", "")).strip():
        bits.append(str(row["genres"]))
    st.caption(" • ".join(bits) if bits else "Movie details are limited in the local dataset")
    st.write(f"**Rating:** {rating_label(row)}")
    if str(row.get("director", "")).strip():
        st.write(f"**Director:** {row['director']}")
    if str(row.get("overview", "")).strip():
        st.write(row["overview"])

if movies.empty:
    st.warning("No movie data found yet. Add CSV files to the data folder, or put MovieLens movies.csv and ratings.csv in ml-latest-small/.")
else:
    st.sidebar.header("1. Choose language")
    language = st.sidebar.selectbox("Movie language", list(LANGUAGES.keys()))
    lang_movies = movies[movies["language"].apply(lambda x: language_matches(x, language))].copy()

    st.sidebar.header("2. Choose a section")
    page = st.sidebar.radio("Go to", ["Browse Movies", "Movie Recommendations", "Movie Details"])

    if lang_movies.empty:
        st.warning(f"No {language} movies are labelled in your local CSV files.")
        st.info("The app intentionally does not guess a movie's language from its title. Add a language column to your catalogue CSV to classify movies accurately.")
    else:
        st.success(f"{len(lang_movies):,} {language} movies found in your local data.")

        if page == "Browse Movies":
            st.subheader(f"Browse {language} Movies")
            query = st.text_input("Search movie title")
            genre_values = sorted({g.strip() for values in lang_movies["genres"] for g in str(values).replace("|", ",").split(",") if g.strip()})
            selected_genre = st.selectbox("Filter by genre", ["All genres"] + genre_values)
            view = lang_movies.copy()
            if query.strip():
                view = view[view["title"].str.contains(query.strip(), case=False, na=False)]
            if selected_genre != "All genres":
                view = view[view["genres"].str.contains(selected_genre, case=False, na=False)]
            st.caption(f"{len(view):,} result(s)")
            if view.empty:
                st.info("No movies match those filters.")
            else:
                for start in range(0, len(view), 10):
                    chunk = view.iloc[start:start+10]
                    cols = st.columns(2)
                    for j, (_, row) in enumerate(chunk.iterrows()):
                        with cols[j % 2]:
                            with st.container(border=True):
                                display_movie(row)

        elif page == "Movie Recommendations":
            st.subheader(f"Recommendations for {language} Movies")
            titles = sorted(lang_movies["title"].unique().tolist(), key=str.casefold)
            selected_title = st.selectbox("Choose a movie you like", titles)
            n_recs = st.slider("Number of recommendations", 3, 15, 5)
            if st.button("Recommend similar movies", type="primary"):
                selected = lang_movies[lang_movies["title"] == selected_title].iloc[0]
                candidates = lang_movies[lang_movies["title"] != selected_title].copy()
                if candidates.empty:
                    st.info("Add more movies in this language to get recommendations.")
                else:
                    # Genre/overview text similarity; no API or external service is used.
                    corpus = (lang_movies["genres"].fillna("") + " " + lang_movies["overview"].fillna("")).tolist()
                    vectorizer = TfidfVectorizer(stop_words="english")
                    try:
                        matrix = vectorizer.fit_transform(corpus)
                        selected_index = lang_movies.index.get_loc(selected.name)
                        scores = cosine_similarity(matrix[selected_index], matrix).flatten()
                        scored = lang_movies.copy()
                        scored["_score"] = scores
                        recs = scored[scored["title"] != selected_title].sort_values("_score", ascending=False).head(n_recs)
                    except ValueError:
                        # If the catalogue has no descriptive text, fall back to genre overlap.
                        selected_genres = set(str(selected["genres"]).replace("|", ",").split(","))
                        scored_rows = []
                        for _, row in candidates.iterrows():
                            other = set(str(row["genres"]).replace("|", ",").split(","))
                            union = selected_genres | other
                            score = len(selected_genres & other) / len(union) if union else 0
                            scored_rows.append((score, row))
                        recs = pd.DataFrame([r for _, r in sorted(scored_rows, key=lambda x: x[0], reverse=True)[:n_recs]])
                    st.markdown(f"**Because you chose:** {selected_title}")
                    for _, row in recs.iterrows():
                        with st.container(border=True):
                            display_movie(row)
                            if "_score" in row and pd.notna(row["_score"]):
                                st.caption(f"Content similarity: {row['_score']:.0%}")

        else:
            st.subheader(f"Movie Details — {language}")
            title = st.selectbox("Select a movie", sorted(lang_movies["title"].unique().tolist(), key=str.casefold))
            row = lang_movies[lang_movies["title"] == title].iloc[0]
            display_movie(row)

st.divider()
with st.expander("How to add movies and ratings"):
    st.markdown("""
**This app does not use TMDB or any API.** It only reads local CSV files.

Create a folder named `data` next to this Python file and add one or more CSV files with these columns:

- `title` — required
- `language` — required for language filtering; use `English`, `Telugu`, `Hindi`, or `Tamil`
- `genres` — optional; examples: `Action,Drama` or `Action|Drama`
- `rating` — optional numeric rating, ideally on a 0–10 scale
- `rating_count` — optional number of ratings
- `year`, `overview`, `director` — optional

Example row: `RRR,Telugu,Action|Drama,8.0,1200,2022,An action drama, S. S. Rajamouli`

MovieLens `movies.csv` and `ratings.csv` are supported, but MovieLens genres do not identify language. Those rows are intentionally not assigned a language automatically. For broad English/Telugu/Hindi/Tamil coverage, provide a language-labelled catalogue CSV from a source you are allowed to use.
""")
