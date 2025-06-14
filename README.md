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
1) credits
2) movies
   ##credits
   Total Records: 4803
   Columns: 4
   Key Information:
   
   | Column     | Description                                                            |
   | ---------- | ---------------------------------------------------------------------- |
   | `movie_id` | Unique identifier for the movie (used to join with the movies dataset) |
   | `title`    | Title of the movie                                                     |
   | `cast`     | JSON string listing cast members with roles and actor names            |
   | `crew`     | JSON string listing crew members like director, writer, producer, etc. |

   ##movies
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
### Data preprocessing is the process of cleaning, transforming, and organizing raw data into a format that can be efficiently used for analysis, machine learning, or data-driven decision-making.

