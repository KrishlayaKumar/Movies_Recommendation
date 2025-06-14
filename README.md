# Movies_Recommendation_system
This is a ML project use for recommending movies based on the movie that you will be providing.

## Steps :-
1) Data collection
2) Data preprocessing
3) Vectorization
4) main function
5) frontend
6) flask

## Data_collection:-
For data we are using **[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)**
This dataset contain 2 CSV file
1) credits:-
   
   Total Records: 4803
   Columns: 4
   Key Information:-
   
   | Column     | Description                                                            |
   | ---------- | ---------------------------------------------------------------------- |
   | `movie_id` | Unique identifier for the movie (used to join with the movies dataset) |
   | `title`    | Title of the movie                                                     |
   | `cast`     | JSON string listing cast members with roles and actor names            |
   | `crew`     | JSON string listing crew members like director, writer, producer, etc. |

2) movies:-
   
   Total Records: 4803
   Columns: 20
   Key Information:

   | Column                                                             | Description                                                      |
   | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
   | `id` / `title`                                                     | Movie ID and title (can be joined with `credits` via `movie_id`) |
   | `budget`                                                           | Budget in USD                                                    |
   | `genres`                                                           | List of genres in JSON format                                    |
   | `keywords`                                                         | List of keywords in JSON format                                  |
   | `overview`                                                         | Brief summary of the movie                                       |
   | `popularity`                                                       | Popularity score                                                 |
   | `release_date`                                                     | Date of release                                                  |
   | `revenue`                                                          | Revenue in USD                                                   |
   | `runtime`                                                          | Duration of the movie in minutes                                 |
   | `vote_average`                                                     | Average rating                                                   |
   | `vote_count`                                                       | Number of votes                                                  |
   | `production_companies`, `production_countries`, `spoken_languages` | JSON lists of associated data                                    |
   | `original_language`, `status`, `tagline`, etc.                     | Additional metadata                                              |


## Data_preprocessing
**Data preprocessing is the process of cleaning, transforming, and organizing raw data into a format that can be efficiently used for analysis, machine learning, or data-driven decision-making.**
A) now we have to choose which feature we have to use/which is more usefull for Movie Recommender

    1)genres 
    2)id 
    3)keywords 
    4)title 
    5)overview
    6)cast
    7)crew
    
B) we will make a tag in this 
   the tag will contain **overview,crew,cast,keywords,genres**
   
C) now we will convert all the 5 col which is going to become our TAG into the correct format
   1) genres :-
      '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
      convert this into dictionary list into list use **ast** and the function is **.literal_eval()**
      code:-
         import ast
         def convert(obj):
             L = []
             for i in ast.literal_eval(obj):
                 L.append(i['name'])
             return L
       **we will convert this into Ex:- [Action, Adventure, Fantasy, Science Fiction]**
      
   2) keywords :-
      same method use in **genres**
      Ex:-[culture clash, future, space war, space colon...	]

   3) Cast:-
      we will tak top 4 Actor from the film
      
      code:-
      import ast
      def convert3(obj):
          L = []
          counter = 0
          for i in ast.literal_eval(obj):
              if counter != 4:
                  L.append(i['name'])
                  counter+=1
              else:
                  break
          return L

      Ex:- [Sam Worthington, Zoe Saldana, Sigourney Weave...]
      
   4) Crew:-
      IN crew we have multiple option like producers, directors, cinematographers, sound mixers, art directors, costume designers, and many more.
      but we will only take directors of the movie.
      
      code:-
      
      def F_director(obj):
       L = []
       for i in ast.literal_eval(obj):
           if i['job'] == "Director":
               L.append(i["name"])
               break
       return L
d) we will convet overview into the list
   code:-
   movies['overview']=movies['overview'].apply(lambda x:x.split())

e) we will apply transformation into the TAG, we will remove the " " space from each word to avoid the confusion between two name
   code:-
   movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
   movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
   movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])

f) add all of them in one
    code:-
    movies['tags'] =movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new dataset will be
| id     | title                                    | tags                                                                                         |
| ------ | ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| 19995  | Avatar                                   | \[In, the, 22nd, century,, a, paraplegic, Marine, is, dispatched, to, the, moon, Pandora...] |
| 285    | Pirates of the Caribbean: At World's End | \[Captain, Barbossa,, long, believed, to, be, dead, has, come, back, to, life...]            |
| 206647 | Spectre                                  | \[A, cryptic, message, from, Bond’s, past, sends, him, on, a, trail, to, uncover...]         |
| 49026  | The Dark Knight Rises                    | \[Following, the, death, of, District, Attorney, Harvey, Dent,, Batman, assumes...]          |
| 49529  | John Carter                              | \[John, Carter, is, a, war-weary,, former, military, captain, who, is, inexplicably...]      |

**convert the list into the string**:-
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

**convert into lowercase**:-
new['tags']=new['tags'].apply(lambda x:x.lower())


## Vectorization
**Vectorization in machine learning and natural language processing (NLP) refers to the process of converting text or categorical data into numerical format (vectors), which models can understand and process.**

code:-
1)
   from sklearn.feature_extraction.text import CountVectorizer
   cv = CountVectorizer(max_features=5000, stop_words='english')
   vectors = cv.fit_transform(new['tags']).toarray()
   
   Purpose: Convert the tags column (which combines plot, cast, genre, etc.) into a numerical matrix.
   max_features=5000: Only the top 5000 most frequent words are considered.
   stop_words='english': Removes common English stopwords (like the, is, and).
   vectors: Now a numerical array where each row corresponds to a movie.

2) Token Stemming:-

   from nltk.stem.porter import PorterStemmer
   ps = PorterStemmer()
   
   def stem(text):
       y = []
       for i in text.split():
           y.append(ps.stem(i))
       return " ".join(y)
   
   new['tags'] = new['tags'].apply(stem)
   
   Purpose: Reduce words to their base/root form.
   Example: "loving" becomes "love".
   Why? It helps reduce dimensionality and treat similar words as the same feature.

3) Re-vectorize After Stemming:-

   cv = CountVectorizer(max_features=5000, stop_words='english')
   vectors = cv.fit_transform(new['tags']).toarray()
   Reason: Stemming changed the tags, so you re-run CountVectorizer to update the vectors accordingly.

4) Similarity Matrix :-
   
   from sklearn.metrics.pairwise import cosine_similarity
   similarity = cosine_similarity(vectors)
      
 5) Saving the Model:-
    
    import pickle
    pickle.dump(new, open('movies.pkl', 'wb'))
    pickle.dump(new, open('similarity.pkl', 'wb'))

## Main function:-
   5) Recommendation Function

   def recommend(movie):
       movie_index = new[new['title'] == movie].index[0]
       distance = similarity[movie_index]
       movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
       for i in movies_list:
           print(new.iloc[i[0]].title)
   
   Ex:- recommend('Avatar')
   
   output:- 
      Aliens vs Predator: Requiem
      Aliens
      Independence Day
      Falcon Rising
      Titan A.E.

 ## Frontend:-

   we will use 2 things
   **HTML**
   **CSS**

## backend:-
   we will use **Flask**


# 🎬 Movie Recommendation System

A Machine Learning-based movie recommender system that suggests similar movies based on the input movie provided by the user.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [1. Data Collection](#1-data-collection)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Vectorization](#3-vectorization)
  - [4. Recommendation Engine](#4-recommendation-engine)
  - [5. Frontend](#5-frontend)
  - [6. Backend (Flask)](#6-backend-flask)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Credits](#credits)

---

## 📖 Overview

This project builds a content-based movie recommendation system using metadata such as genres, cast, crew, and keywords. When a user enters a movie title, the system recommends five similar movies based on their textual descriptions.

---

## 📂 Dataset

The system uses the **[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)** which contains two main CSV files:

### 1. `tmdb_5000_credits.csv`

| Column     | Description                                                            |
|------------|------------------------------------------------------------------------|
| `movie_id` | Unique movie ID (used to join with the movies dataset)                |
| `title`    | Title of the movie                                                     |
| `cast`     | JSON string listing actors and roles                                   |
| `crew`     | JSON string listing directors, writers, and other crew members         |

### 2. `tmdb_5000_movies.csv`

| Column             | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `id`, `title`      | Movie ID and title                                            |
| `genres`           | List of genres (JSON format)                                  |
| `keywords`         | Keywords associated with the movie (JSON format)              |
| `overview`         | Plot summary                                                  |
| `cast`, `crew`     | Enriched via the credits dataset                              |
| `budget`, `revenue`, `runtime`, `popularity`, `vote_average`, etc. | Additional metadata |

---

## 🔄 Project Workflow

### 1. Data Collection

The raw data is merged on the `movie_id` field from both datasets, combining movie metadata and credits.

---

### 2. Data Preprocessing

Preprocessing is key to extracting meaningful tags for comparison.

#### ➤ Selected Features:
- `genres`
- `keywords`
- `overview`
- `cast` (top 4 actors)
- `crew` (director only)

#### ➤ Preprocessing Steps:
- **Parse JSON columns** using Python’s `ast.literal_eval`.
- **Extract useful tokens** from lists and keep only important names.
- **Clean and normalize text**:
  - Remove spaces between multi-word names.
  - Convert everything to lowercase.
  - Combine all selected fields into a single `tags` column.

**Final Structure:**

| id     | title     | tags                                                          |
|--------|-----------|---------------------------------------------------------------|
| 19995  | Avatar    | in the 22nd century a paraplegic marine ... jamescameron ...  |

---

### 3. Vectorization

We use `CountVectorizer` to convert text tags into numerical vectors.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()

