from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

# Function to download from Google Drive using file ID
def download_from_gdrive(file_id, dest_path):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(URL)
    with open(dest_path, "wb") as f:
        f.write(response.content)

# Ensure current file's directory is used for path handling
base_path = os.path.dirname(__file__)
movies_path = os.path.join(base_path, "movies.pkl")
similarity_path = os.path.join(base_path, "similarity.pkl")

# Download files only if they don't exist
if not os.path.exists(movies_path):
    download_from_gdrive("1qCYhr-p3oS8Fk9PewKU7Hh1F26n8g2Uf", movies_path)

if not os.path.exists(similarity_path):
    download_from_gdrive("1581DyhbQTtGXfLJ33g0uIG23J4qWGUzG", similarity_path)

# Load data
movies = pickle.load(open(movies_path, "rb"))
similarity = pickle.load(open(similarity_path, "rb"))

# TMDB API Key
TMDB_API_KEY = "080e6d69f3f04d9d92a0142b14c81d5d"

def fetch_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url, timeout=3)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return "https://via.placeholder.com/300x450?text=No+Image"

def recommend(movie_title):
    movie_title = movie_title.lower()
    titles = movies['title'].str.lower()

    if movie_title not in titles.values:
        return []

    index = titles[titles == movie_title].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        title = movies.iloc[i[0]].title
        poster_url = fetch_poster(title)
        recommendations.append({"title": title, "poster": poster_url})
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    movie = ""
    if request.method == 'POST':
        movie = request.form.get('movie')
        if movie:
            recommendations = recommend(movie)
    return render_template('index.html', movie=movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
