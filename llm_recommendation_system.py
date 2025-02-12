# Import core components from the Surprise library for collaborative filtering,
# including dataset handling, rating scale processing, and the SVD algorithm.
from surprise import Dataset, Reader, SVD, KNNBasic, SVDpp, accuracy
from tabulate import tabulate # For displaying the results matrix

# Import model selection utilities from Surprise for train-test splits and cross-validation.
from surprise.model_selection import train_test_split, cross_validate

# pandas is used for data manipulation and analysis (e.g., reading CSVs, merging DataFrames).
import pandas as pd

# Regular expressions for extracting similarity scores or parsing strings in LLM responses.
import re

# The requests library is used for handling HTTP API requests to the LLM service.
import requests

# The IMDbPY library (Cinemagoer) is used to access IMDb data for movies.
from imdb import Cinemagoer, IMDbError

# The math module is imported for additional mathematical operations (e.g., rounding, clamping).
import math

# The logging module is used to configure logging, such as suppressing noisy logs from IMDbPY.
import logging

# Configure logging for the IMDbPY library to suppress detailed output.
logging.getLogger('imdbpy').setLevel(logging.ERROR)
logging.getLogger('imdbpy').disabled = True

# The time module provides functions to measure or delay execution (e.g., retry delays).
import time

# The random module is used to select a random movie rating to drop (leave-one-out testing).
import random

# argparse is used to parse command-line arguments for configuring running mode and dataset choices.
import argparse

# os is used for operating system interfaces such as file system operations (e.g., checking paths, creating directories).
import os

# datetime is used to handle and manipulate dates, such as determining movie release date ranges.
import datetime

# The typing module provides support for type hints throughout the code.
from typing import Any, Dict, List, Optional, Tuple, Union

def normalize_popularity_score(votes: int | float | None, max_rank: int = 1_000_000) -> int | None:
    """
    Convert raw vote counts to a standardized popularity score on a 0–100 scale using linear interpolation and banker's rounding.

    The function computes a raw percentage as:
      raw_percentage = 100 * (votes / max_rank)
    indicating the proportion of the maximum expected vote count (max_rank). This raw percentage can exceed 100 
    when votes are greater than max_rank. However, the function then applies banker's rounding and clamps the result
    to ensure that the final popularity score is within the range 0 to 100. Essentially, any vote count that meets or
    exceeds max_rank will yield a final score of 100.

    Parameters:
      votes (int | float | None):
          The raw vote count for a movie. Only positive values are considered valid; if votes is None or non-positive, the function returns None.
      
      max_rank (int, default=1_000_000):
          The threshold vote count corresponding to a maximum popularity score of 100. Vote counts equal to or greater 
          than this value will result in a score of 100, while values below this threshold are scaled proportionally.

    Returns:
      int | None:
          - Returns 100 if votes is greater than or equal to max_rank.
          - Returns an integer between 0 and 99 if votes is positive but less than max_rank.
          - Returns None if votes is None or non-positive.

    Examples:
      >>> normalize_popularity_score(1_250_000)
      100
      >>> normalize_popularity_score(750_000)
      75
      >>> normalize_popularity_score(123_456)
      12
    """
    # Validate input: Only positive vote counts are valid.
    if votes is None or votes <= 0:
        return None
    
    # Calculate the popularity ratio relative to the maximum rank.
    popularity_ratio = votes / max_rank
    
    # Convert the ratio to a percentage scale.
    raw_percentage = 100 * popularity_ratio
    
    # Apply banker's rounding (round-half-to-even) to obtain the nearest integer.
    rounded_score = round(raw_percentage)
    
    # Clamp the result to ensure it doesn't exceed 100.
    clamped_score = max(0, min(rounded_score, 100))
    
    return clamped_score

def calculate_cumulative_hit_rate(
    algo: Any,
    ratings_dataframe: pd.DataFrame,
    id_to_title: Dict[Any, str],
    combined_dataframe: pd.DataFrame,
    model_name: str,
    chat_format: str,
    descriptions_path: str,
    api_url: str,
    headers: Dict[str, str],
    preferences_path: str,
    scores_path: str,
    n: int = 10,
    threshold: float = 4.0,
    user_ids: Optional[List[Any]] = None,
    use_llm: bool = False,
    max_retries: int = 5,
    delay_between_attempts: int = 1,
    num_favorites: int = 3,
    search_count: Optional[int] = None
) -> Tuple[float, Optional[float]]:
    """
    Calculate the cumulative hit rate of a movie recommendation system using leave-one-out cross-validation.

    This function evaluates a recommendation algorithm by testing if a movie that a user has rated 
    highly (i.e., a rating greater than or equal to the specified threshold) appears in the top N 
    recommendations when that movie is left out during training. In addition, when `use_llm` is True, 
    the base recommendations are further refined using an LLM-based process that considers user preferences, 
    movie descriptions, and popularity scores.

    Cross-Validation Details:
      - For each user (or for the subset provided in `user_ids`), one random rating (if the user has more than one)
        is removed from the ratings DataFrame to form a test set.
      - The algorithm is trained on the remaining ratings.
      - A hit is counted if the left-out movie appears among the top N recommendations.

    Parameters:
      algo (Any):
          A trained recommendation algorithm from the Surprise library (e.g., an SVD instance) that implements 
          a .fit(trainset) method and a .predict(user_id, movie_id) method.
      ratings_dataframe (pd.DataFrame):
          DataFrame containing all user ratings with at least the columns ['userId', 'movieId', 'rating'].
      id_to_title (Dict[Any, str]):
          A dictionary mapping each movie ID to its corresponding movie title.
      combined_dataframe (pd.DataFrame):
          DataFrame containing movie metadata (e.g., IMDb IDs) required for LLM processing.
      model_name (str):
          The name of the language model used for generating or refining movie descriptions.
      chat_format (str):
          A prompt format string for the LLM that outlines how few-shot examples and movie title are formatted.
      descriptions_path (str):
          File path to a CSV file containing pre-generated movie descriptions.
      api_url (str):
          The URL endpoint for processing movie descriptions via the LLM.
      headers (Dict[str, str]):
          A dictionary of HTTP headers for API requests (e.g., for authorization).
      preferences_path (str):
          File path to a CSV file containing user preferences (such as prioritizing high ratings or popularity).
      scores_path (str):
          File path to a CSV file containing movie scores (IMDb ratings, raw and normalized popularity).
      n (int, default=10):
          The number of top recommendations to consider for each test case.
      threshold (float, default=4.0):
          The minimum rating for a movie to be regarded as relevant when testing.
      user_ids (Optional[List[Any]], default=None):
          A list of specific user IDs to evaluate. If None, the evaluation is performed on all users.
      use_llm (bool, default=False):
          Flag indicating whether to apply LLM-based refinement on the base recommendations.
      max_retries (int, default=5):
          The maximum number of retries when fetching movie descriptions or related data via the API.
      delay_between_attempts (int, default=1):
          The number of seconds to wait between each retry attempt.
      num_favorites (int, default=3):
          The number of a user's favorite movies to consider when calculating similarity during LLM-enhanced refinement.
      search_count (Optional[int], default=None):
          The number of top recommendations to fetch initially from the base algorithm before LLM refinement.
          If None, defaults to n * 10.  This allows controlling the size of the candidate set for LLM processing.

    Returns:
      Tuple[float, Optional[float]]:
          A tuple containing:
            - base_hit_rate (float): The fraction of test cases where the withheld movie appears in the top N recommendations 
              using the base recommendation algorithm.
            - llm_hit_rate (Optional[float]): The fraction for LLM-enhanced recommendations if use_llm is True; otherwise, None.
    """
    # Create a copy of the ratings DataFrame (this copy will have the test samples removed)
    ratings_dataframe_testset_removed = ratings_dataframe.copy()
    
    # Initialize the leave-one-out test set list.
    loo_testset = []

    # Determine unique users to evaluate; if user_ids is None, use all user IDs.
    if user_ids is None:
        unique_user_ids = ratings_dataframe_testset_removed['userId'].unique()
    else:
        unique_user_ids = user_ids

    # Create the leave-one-out test set by removing one random rating per user (if the user has >1 rating).
    for user_id in unique_user_ids:
        user_ratings = ratings_dataframe_testset_removed[ratings_dataframe_testset_removed['userId'] == user_id]
        if len(user_ratings) > 1:
            test_rating_index = random.randint(0, len(user_ratings) - 1)
            test_rating = user_ratings.iloc[test_rating_index]
            loo_testset.append((user_id, test_rating['movieId'], test_rating['rating']))
            ratings_dataframe_testset_removed = ratings_dataframe_testset_removed.drop(user_ratings.index[test_rating_index])
    
    # Define a Reader with the rating scale (0.5, 5.0)
    reader = Reader(rating_scale=(0.5, 5.0))
    
    # Build the training set from the remaining ratings.
    data = Dataset.load_from_df(ratings_dataframe_testset_removed[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    # Train the algorithm on the training set.
    algo.fit(trainset)
    
    # Retrieve all movie IDs to serve as candidates for recommendations.
    all_movie_ids = ratings_dataframe['movieId'].unique()
    
    # Initialize counters for base and LLM-enhanced hits, and the total number of relevant test cases.
    base_hit_count = 0
    llm_hit_count = 0 
    total_count = 0
    
    # Load user preferences and movie scores (required for LLM-enhanced refinement).
    max_user_id_val = ratings_dataframe['userId'].max()
    preferences_df = load_user_preferences(preferences_path, max_user_id_val)
    movie_scores_df = load_movie_scores(scores_path, combined_dataframe['movieId'].max())
    
    # Evaluate each test case from the leave-one-out set.
    for user_id, movie_id, rating in loo_testset:
        # Only consider test cases where the rating meets the threshold.
        if rating >= threshold:
            total_count += 1
            
            # Base recommendations: Get top N recommendations for the user.
            top_n_recommendations = get_top_n_recommendations(
                algo, user_id, all_movie_ids, ratings_dataframe_testset_removed, id_to_title, n
            )
            recommended_movie_ids = [rec_movie_id for rec_movie_id, _, _ in top_n_recommendations]
            
            # Count a base hit if the left-out movie appears in the recommendations.
            if movie_id in recommended_movie_ids:
                base_hit_count += 1
            
            # If using LLM-enhanced recommendations, proceed with further refinement.
            if use_llm:

                # Determine the number of movies to fetch for LLM processing
                num_movies_for_llm = search_count if search_count is not None else n * 10

                # Fetch an extended candidate set for further processing.
                top_n_times_x_for_user = get_top_n_recommendations(
                    algo, user_id, all_movie_ids, ratings_dataframe_testset_removed, id_to_title, num_movies_for_llm
                )
                
                # Retrieve user-specific preferences: ratings, popularity, the overall preference text, and date range.
                prioritize_ratings = preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_ratings'].iloc[0]
                prioritize_popular = preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_popular'].iloc[0]
                user_preferences = preferences_df.loc[preferences_df['userId'] == user_id, 'preferences'].iloc[0]
                user_date_range = preferences_df.loc[preferences_df['userId'] == user_id, 'date_range'].iloc[0]
                
                # Get movie descriptions via the LLM process.
                movie_descriptions = get_movie_descriptions(
                    top_n_times_x_for_user,
                    combined_dataframe,
                    model_name,
                    chat_format,
                    descriptions_path,
                    scores_path,
                    api_url,
                    headers,
                    max_retries,
                    delay_between_attempts
                )
                
                # Retrieve the user's favorite movies.
                favorite_movie_titles = get_user_favorite_movies(
                    user_id, ratings_dataframe_testset_removed, id_to_title, num_favorites=num_favorites
                )
                
                # Use the LLM to select the top N movies most similar to the user's preferences.
                top_n_similiar_movies = find_top_n_similar_movies(
                    user_preferences,
                    movie_descriptions,
                    id_to_title,
                    model_name,
                    chat_format,
                    n,
                    api_url,
                    headers,
                    movie_scores_df,
                    prioritize_ratings=prioritize_ratings,
                    prioritize_popular=prioritize_popular, 
                    favorite_movies=favorite_movie_titles,
                    num_favorites=num_favorites,
                    date_range=user_date_range
                )
                llm_recommended_movie_ids = [rec_movie_id for rec_movie_id, _ in top_n_similiar_movies]
                
                # Count an LLM-enhanced hit if the left-out movie is among those recommendations.
                if movie_id in llm_recommended_movie_ids:
                    llm_hit_count += 1
    
    # Calculate the hit rate for the base algorithm and, if applicable, for the LLM-enhanced version.
    base_hit_rate = base_hit_count / total_count if total_count > 0 else 0
    llm_hit_rate = (llm_hit_count / total_count) if total_count > 0 else 0
    
    return base_hit_rate, llm_hit_rate if use_llm else None

def load_movie_scores(scores_path: str, max_movie_id: int) -> pd.DataFrame:
    """
    Load movie scores from a CSV file and ensure that every movie ID from 1 to max_movie_id has an entry.
    
    This function attempts to read a CSV file at scores_path that contains movie scores and metadata.
    If the file exists, it replaces any missing or placeholder values (such as empty strings, zeros,
    or NaN values) with None. It then creates a complete DataFrame that contains every movie ID in the
    range [1, max_movie_id], merging the existing data with newly initialized rows where data is missing.
    If the file does not exist, it creates a new DataFrame with movieId values from 1 to max_movie_id and
    initializes all other columns with None.
    
    Parameters:
      scores_path (str):
          The file path to the CSV file containing movie scores.
      max_movie_id (int):
          The maximum MovieLens ID to accommodate. All movie IDs from 1 to this number will be present
          in the returned DataFrame.
    
    Returns:
      pd.DataFrame:
          A DataFrame with the following columns:
            - movieId: MovieLens ID (int)
            - imdbId: IMDb ID (or None if unknown)
            - imdb_rating: IMDb rating on a 0-10 scale (or None if unavailable)
            - normalized_popularity: Popularity score scaled between 0 and 100 (or None if unavailable)
            - raw_popularity: Raw IMDb popularity rank (or None if unavailable)
    """
    if os.path.exists(scores_path):
        # Read the CSV file containing movie score data.
        movie_scores = pd.read_csv(scores_path)
        
        # Replace placeholders and missing values with None.
        movie_scores = movie_scores.replace({
            '': None,
            0: None,
            'nan': None,
            float('nan'): None
        })
        
        # Create a DataFrame with all movie IDs from 1 to max_movie_id.
        all_movie_ids = pd.DataFrame({'movieId': range(1, max_movie_id + 1)})
        
        # Merge the full list of movie IDs with the loaded movie scores.
        complete_scores = pd.merge(all_movie_ids, movie_scores, on='movieId', how='left')
        
        # Replace any remaining NaN values with None.
        complete_scores = complete_scores.where(pd.notnull(complete_scores), None)
    else:
        # If the CSV does not exist, initialize a new DataFrame with all required columns set to None.
        complete_scores = pd.DataFrame({
            'movieId': range(1, max_movie_id + 1),
            'imdbId': [None] * max_movie_id,
            'imdb_rating': [None] * max_movie_id,
            'normalized_popularity': [None] * max_movie_id,
            'raw_popularity': [None] * max_movie_id
        })
    
    return complete_scores

def save_movie_scores(scores_df: pd.DataFrame, scores_path: str) -> None:
    """
    Save movie scores DataFrame to a CSV file.

    This function ensures that the directory for the provided file path exists (creating it if necessary) 
    and then writes the DataFrame to a CSV file without including the DataFrame index. The movie scores 
    DataFrame is expected to contain fields such as movie identifiers, IMDb ratings, normalized popularity 
    scores, and raw popularity counts.

    Parameters:
      scores_df (pd.DataFrame): DataFrame containing movie scores. Expected columns include:
                                - 'movieId': MovieLens ID for the movie.
                                - 'imdbId': IMDb identifier.
                                - 'imdb_rating': The IMDb rating (0-10).
                                - 'normalized_popularity': Normalized popularity score (0-100).
                                - 'raw_popularity': Raw vote/popularity count.
      scores_path (str): The full file path (including the directory and file name) where the CSV file will be saved.

    Returns:
      None

    Functionality:
      - Ensures that the directory specified in scores_path exists by creating it if it does not.
      - Saves the scores_df DataFrame to the specified path in CSV format without writing the DataFrame index.
      - If an IOError or OSError occurs during directory creation or file writing, the error is caught and 
        an error message is printed.
    """
    try:
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        scores_df.to_csv(scores_path, index=False)
    except (IOError, OSError) as e:
        print(f"Error saving movie scores: {e}")

def create_movie_mappings(movies_df: pd.DataFrame) -> Tuple[Dict[Any, str], Dict[str, Any]]:
    """
    Create mapping dictionaries between MovieLens movie IDs and movie titles.

    This function expects a pandas DataFrame with at least the following two columns:
      - 'movieId': A unique identifier for each movie (typically an integer).
      - 'title': The title of the movie (a string).

    It returns:
      - id_to_title: A dictionary mapping each movieId to its corresponding movie title.
      - title_to_id: A dictionary mapping each movie title to its corresponding movieId.

    Example:
        Suppose movies_df contains:
            movieId    title
                1    Toy Story (1995)
                2    Jumanji (1995)

        Then,
            id_to_title = {1: "Toy Story (1995)", 2: "Jumanji (1995)"}
            title_to_id = {"Toy Story (1995)": 1, "Jumanji (1995)": 2}

    Parameters:
      movies_df (pd.DataFrame): DataFrame containing movie information with 'movieId' and 'title' columns.

    Returns:
      Tuple[Dict[Any, str], Dict[str, Any]]:
        - id_to_title: Dictionary mapping movieId to title.
        - title_to_id: Dictionary mapping title to movieId.
    """
    # Create a dictionary mapping movieId to title
    id_to_title = pd.Series(movies_df.title.values, index=movies_df.movieId).to_dict()

    # Create a dictionary mapping title to movieId
    title_to_id = pd.Series(movies_df.movieId.values, index=movies_df.title).to_dict()

    return id_to_title, title_to_id

def create_id_mappings(links_df: pd.DataFrame) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    Create mapping dictionaries between MovieLens movie IDs and IMDb IDs.

    This function expects a pandas DataFrame that contains at least the following columns:
      - 'movieId': A unique identifier for each movie in the MovieLens dataset (typically an integer).
      - 'imdbId': The corresponding IMDb identifier for each movie (which may be represented as a string or numeric value).

    It returns two dictionaries:
      - movielens_to_imdb: A dictionary mapping each MovieLens movieId to its corresponding IMDb ID.
      - imdb_to_movielens: A dictionary mapping each IMDb ID to its corresponding MovieLens movieId.

    Example:
        Given a DataFrame `links_df` as follows:
           movieId    imdbId
           1          0114709
           2          0113497

        The function will return:
           movielens_to_imdb = {1: "0114709", 2: "0113497"}
           imdb_to_movielens = {"0114709": 1, "0113497": 2}

    Parameters:
      links_df (pd.DataFrame): A DataFrame containing movie mapping information with columns 'movieId' and 'imdbId'.

    Returns:
      Tuple[Dict[Any, Any], Dict[Any, Any]]:
        - movielens_to_imdb: Dictionary mapping MovieLens movieId to IMDb ID.
        - imdb_to_movielens: Dictionary mapping IMDb ID to MovieLens movieId.
    """
    # Create a dictionary mapping MovieLens movieId to IMDb ID
    movielens_to_imdb = pd.Series(links_df.imdbId.values, index=links_df.movieId).to_dict()

    # Create a dictionary mapping IMDb ID to MovieLens movieId
    imdb_to_movielens = pd.Series(links_df.movieId.values, index=links_df.imdbId).to_dict()

    return movielens_to_imdb, imdb_to_movielens

def get_top_n_recommendations(
    algo: Any,
    user_id: Any,
    all_movie_ids: List[Any],
    ratings_dataframe: pd.DataFrame,
    id_to_title: Dict[Any, str],
    n: int = 10
) -> List[Tuple[Any, str, float]]:
    """
    Generate the top N movie recommendations for a given user.

    This function works by:
      1. Identifying the set of movie IDs that the user has already rated from the ratings_dataframe.
      2. For each movie in all_movie_ids that the user has not rated, it uses the trained recommendation algorithm
         (algo) to predict an estimated rating via its .predict(user_id, movie_id) method.
      3. Sorting these predictions in descending order by the estimated rating value.
      4. Returning the top N predictions as a list of tuples, where each tuple contains:
         (movieId, movie title, estimated rating).

    Parameters:
      algo (Any):
          A trained recommendation algorithm (for example, an instance from the Surprise library) that implements a
          .predict(user_id, movie_id) method returning an object with an attribute .est representing the estimated rating.
      user_id (Any):
          The identifier of the user (typically an int or str) for whom the recommendations are generated.
      all_movie_ids (List[Any]):
          A list of movie IDs available in the dataset. These IDs are typically integers or strings.
      ratings_dataframe (pd.DataFrame):
          A pandas DataFrame containing user ratings with at least the columns ['userId', 'movieId']. This is used to filter
          out movies that the user has already rated.
      id_to_title (Dict[Any, str]):
          A dictionary that maps movie IDs to their corresponding movie titles. The keys are movie IDs and the values are strings.
      n (int, default=10):
          The number of top recommendations to return.

    Returns:
      List[Tuple[Any, str, float]]:
          A list of up to n tuples where each tuple consists of:
          - The movieId of the recommended movie.
          - The movie title corresponding to the movieId (as obtained from id_to_title; if not found, defaults to "Unknown").
          - The estimated rating (a float) predicted by the recommendation algorithm for that movie.
    """
    # Identify movies already rated by the user.
    rated_movies = set(ratings_dataframe[ratings_dataframe['userId'] == user_id]['movieId'])
    
    # For each movie the user hasn't rated, predict the rating using the recommendation algorithm.
    predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movies]
    
    # Sort the predictions in descending order of their estimated ratings.
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Create a list of tuples for the top n predictions:
    # Each tuple contains (movieId, title, estimated rating)
    top_n = [(pred.iid, id_to_title.get(pred.iid, "Unknown"), pred.est) for pred in predictions[:n]]
    
    return top_n

def retrieve_all_descriptions(
    combined_dataframe: pd.DataFrame,
    model_name: str,
    chat_format: str,
    descriptions_path: str,
    scores_path: str,
    url: str,
    headers: dict,
    start_movie_id: int = 1,
    max_retries: int = 5,
    delay_between_attempts: int = 1
) -> pd.DataFrame:
    """
    Retrieve and cache movie descriptions and IMDb scores for all movies.

    This function processes each movie in the combined_dataframe by performing the following steps:
      1. Load existing cached scores and descriptions from local CSV files.
      2. For each movie whose movieId is >= start_movie_id, check whether:
         - The description is missing or empty.
         - The IMDb rating (score) is missing.
      3. If any data is missing, call the get_movie_with_retries() helper, which will:
         - First try to fetch movie details from IMDb (including rating, votes, and plot).
         - On failure, fall back to generating a description via an LLM using the provided model and prompt format.
      4. If IMDb data is retrieved:
         - Update the cached description by extracting the plot (or generating one via LLM if needed).
         - Update the movie’s scores (IMDb rating, raw popularity, and normalized popularity, using normalize_popularity_score()).
      5. If no IMDb data is retrieved and the description is missing, generate a description with few-shot LLM.
      6. Periodically (every 100 movies processed) save the current cached descriptions and scores to their respective CSV files.
      7. At the end, perform a final save and return the updated descriptions DataFrame.

    Parameters:
      combined_dataframe (pd.DataFrame):
        DataFrame containing movie metadata (including columns such as 'movieId', 'title', and 'imdbId').
      model_name (str):
        Name of the language model to use for generating descriptions.
      chat_format (str):
        Prompt format string used by the LLM to generate few-shot descriptions.
      descriptions_path (str):
        File path to the descriptions cache CSV file.
      scores_path (str):
        File path to the IMDb scores cache CSV file.
      url (str):
        LLM API endpoint URL.
      headers (dict):
        HTTP headers for API requests.
      start_movie_id (int, default=1):
        Movie ID from which to start processing (allows resuming from a specific point).
      max_retries (int, default=5):
        Maximum number of retry attempts for API calls (for both IMDb and LLM requests).
      delay_between_attempts (int, default=1):
        Delay in seconds between retry attempts.

    Returns:
      pd.DataFrame:
        A DataFrame containing all movie descriptions (updated cache).

    Notes:
      - Descriptions and scores are saved to cache files every 100 movies processed.
      - A progress message is shown every 10 seconds
      - For IMDb scores, missing values are represented as None rather than 0.
      - Both missing descriptions and missing score data are handled independently.
      - Cached data is used to avoid repeated API calls, improving efficiency.
      - The function uses IMDb API (with get_movie_with_retries()) and, if needed, falls back to LLM generation.
    """
    # Load the cached movie scores and cached descriptions from CSV files
    movie_scores_df = load_movie_scores(scores_path, combined_dataframe['movieId'].max())
    cached_descriptions = load_cached_descriptions(descriptions_path, combined_dataframe['movieId'].max())
    
    # Track progress via timestamps
    last_print_time = time.time()

    # Process each row (movie) in the combined dataframe
    for index, row in combined_dataframe.iterrows():
        current_time = time.time()
        if current_time - last_print_time >= 10:
            print(f"Processed {index+1} movies so far...")
            last_print_time = current_time

        movie_id = row['movieId']
        if movie_id < start_movie_id:
            continue
        
        # Retrieve the existing description and IMDb rating from the cached DataFrames.
        # Explicit check for missing description using empty string comparison.
        cached_description = cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'].iloc[0]
        current_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0]
        needs_scores = current_rating is None

        # If the description is missing or the IMDb rating is missing, we attempt data retrieval.
        if cached_description == "" or needs_scores:
            movie_title = row['title']
            imdb_id = row['imdbId']
            
            # Call helper function that uses retries to fetch movie details (or generate data with fallback)
            movie, rating, popularity = get_movie_with_retries(
                imdb_id, movie_title, model_name, chat_format, url, headers, max_retries, delay_between_attempts
            )
            
            if movie:
                # Update the description if missing in cache:
                if cached_description == "":
                    if 'plot' in movie:
                        description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
                    else:
                        description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                    description = description.replace('\n', ' ').replace('\r', ' ').strip()
                    cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

                # Update IMDb scores if they are missing:
                if needs_scores:
                    normalized_popularity = normalize_popularity_score(popularity) if popularity else None
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'] = imdb_id
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'] = rating if rating is not None else None
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'] = normalized_popularity
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'] = popularity if popularity is not None else None
            
            # If no movie data could be retrieved and description is still missing, generate it via LLM.
            elif cached_description == "":
                description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                description = description.replace('\n', ' ').replace('\r', ' ').strip()
                cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

        # Save cached data every 100 movies processed
        if (index + 1) % 100 == 0:
            save_cached_descriptions(cached_descriptions, descriptions_path)
            save_movie_scores(movie_scores_df, scores_path)

    # Final save after processing all movies
    save_cached_descriptions(cached_descriptions, descriptions_path)
    save_movie_scores(movie_scores_df, scores_path)

    return cached_descriptions

def retrieve_all_descriptions(
    combined_dataframe: pd.DataFrame,
    model_name: str,
    chat_format: str,
    descriptions_path: str,
    scores_path: str,
    url: str,
    headers: dict,
    start_movie_id: int = 1,
    max_retries: int = 5,
    delay_between_attempts: int = 1
) -> pd.DataFrame:
    """
    Retrieve and cache movie descriptions and IMDb scores for all movies.

    This function processes each movie in the combined_dataframe by performing the following steps:
      1. Load existing cached scores and descriptions from local CSV files.
      2. For each movie with movieId >= start_movie_id, check whether:
         - The description is missing or empty.
         - The IMDb rating is missing.
         - EITHER the raw popularity OR the normalized popularity score is missing.
            (Both fields must be present in order to consider popularity data up-to-date.)
      3. If any of the above data is missing, call the get_movie_with_retries() helper, which will:
         - First attempt to fetch movie details from IMDb (including rating, votes, and plot).
         - On failure, fall back to generating a description via an LLM using the provided model and prompt format.
      4. If IMDb data is retrieved:
         - Update the cached description by extracting the movie 'plot' (or by using an LLM if needed).
         - Update the movie’s scores:
             • Update the IMDb rating if it was missing.
             • Update the raw popularity and calculate the normalized popularity using normalize_popularity_score()
               if either popularity field was missing.
      5. If no IMDb data is retrieved and a description is still missing, generate a description with an LLM.
      6. Periodically (every 100 movies processed) save the current cached descriptions and scores to their respective CSV files.
      7. Perform a final save after processing all movies and return the updated descriptions DataFrame.

    Parameters:
      combined_dataframe (pd.DataFrame):
        DataFrame containing movie metadata (including columns such as 'movieId', 'title', and 'imdbId').
      model_name (str):
        Name of the language model to use for generating descriptions.
      chat_format (str):
        Prompt format string used by the LLM to generate few-shot descriptions.
      descriptions_path (str):
        File path to the descriptions cache CSV file.
      scores_path (str):
        File path to the IMDb scores cache CSV file.
      url (str):
        LLM API endpoint URL.
      headers (dict):
        HTTP headers for API requests.
      start_movie_id (int, default=1):
        Movie ID from which to start processing (allows resuming from a specific point).
      max_retries (int, default=5):
        Maximum number of retry attempts for API calls (for both IMDb and LLM requests).
      delay_between_attempts (int, default=1):
        Delay in seconds between retry attempts.

    Returns:
      pd.DataFrame:
        A DataFrame containing all movie descriptions (updated cache).

    Notes:
      - Cached descriptions and scores are saved every 100 movies processed.
      - A progress message is printed every 10 seconds.
      - For IMDb scores, missing values are represented as None.
      - Both missing descriptions and missing score data are handled independently.
      - In the case of popularity data, the function now requires that both the raw popularity and the
        normalized popularity fields exist. If either is None, the popularity data is considered outdated
        and will be updated.
      - The function uses cached data to avoid repeated API calls. It attempts to retrieve missing data via the
        IMDb API (using get_movie_with_retries()) and falls back to LLM generation if necessary.
    """
    # Load the cached movie scores and cached descriptions from CSV files
    movie_scores_df = load_movie_scores(scores_path, combined_dataframe['movieId'].max())
    cached_descriptions = load_cached_descriptions(descriptions_path, combined_dataframe['movieId'].max())
    
    # Track progress using timestamps for periodic messages
    last_print_time = time.time()

    # Process each row (movie) in the combined dataframe
    for index, row in combined_dataframe.iterrows():
        current_time = time.time()
        if current_time - last_print_time >= 10:
            print(f"Processed {index+1} movies so far...")
            last_print_time = current_time

        movie_id = row['movieId']
        if movie_id < start_movie_id:
            continue
        
        # Retrieve cached description and IMDb rating/popularity
        cached_description = cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'].iloc[0]
        current_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0]
        current_raw_popularity = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'].iloc[0]
        current_normalized = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'].iloc[0]
        
        # Check separately for missing IMDb rating and missing popularity data
        needs_rating = current_rating is None
        needs_popularity = (current_raw_popularity is None) or (current_normalized is None)

        # If the description is missing or if either rating or popularity data is missing, attempt retrieval.
        if cached_description == "" or needs_rating or needs_popularity:
            movie_title = row['title']
            imdb_id = row['imdbId']
            
            # Attempt to fetch movie details via a helper that implements retries,
            # falling back on LLM generation if necessary.
            movie, rating, popularity = get_movie_with_retries(
                imdb_id, movie_title, model_name, chat_format, url, headers, max_retries, delay_between_attempts
            )
            
            if movie:
                # If description is missing in the cache, update it.
                if cached_description == "":
                    if 'plot' in movie:
                        description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
                    else:
                        description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                    description = description.replace('\n', ' ').replace('\r', ' ').strip()
                    cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

                # Update IMDb rating if missing.
                if needs_rating:
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'] = imdb_id
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'] = rating if rating is not None else None
                
                # Update popularity information if either raw or normalized popularity is missing.
                if needs_popularity:
                    normalized_popularity = normalize_popularity_score(popularity) if popularity else None
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'] = normalized_popularity
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'] = popularity if popularity is not None else None
            
            # If no IMDb data is retrieved and the description is still missing, generate a description with LLM.
            elif cached_description == "":
                description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                description = description.replace('\n', ' ').replace('\r', ' ').strip()
                cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

        # Save cached data every 100 movies processed.
        if (index + 1) % 100 == 0:
            save_cached_descriptions(cached_descriptions, descriptions_path)
            save_movie_scores(movie_scores_df, scores_path)

    # Final save after processing all movies.
    save_cached_descriptions(cached_descriptions, descriptions_path)
    save_movie_scores(movie_scores_df, scores_path)

    return cached_descriptions

def retrieve_all_descriptions(
    combined_dataframe: pd.DataFrame,
    model_name: str,
    chat_format: str,
    descriptions_path: str,
    scores_path: str,
    url: str,
    headers: dict,
    start_movie_id: int = 1,
    max_retries: int = 5,
    delay_between_attempts: int = 1
) -> pd.DataFrame:
    """
    Retrieve and cache movie descriptions and IMDb scores for all movies.

    This function processes each movie in the combined_dataframe by performing the following steps:
      1. Loads existing cached scores and descriptions from local CSV files.
      2. For each movie with movieId >= start_movie_id, it checks whether:
         - The description is missing or empty.
         - The IMDb rating is missing.
         - EITHER the raw popularity OR the normalized popularity score is missing 
           (both fields must be non-None for popularity data to be considered complete).
         - The cached IMDb ID is missing.
      3. If any of the above data is missing, it calls the get_movie_with_retries() helper, which:
         - Attempts to fetch movie details (including rating, votes, and plot) from IMDb.
         - On failure, falls back to generating a description via an LLM using the provided model and prompt format.
      4. If IMDb data is retrieved:
         - Updates the cached description by extracting the movie’s 'plot' (or by using an LLM if necessary).
         - Updates the movie’s scores:
             • Sets the IMDb rating if it was missing.
             • Sets the raw popularity, and calculates the normalized popularity using normalize_popularity_score() if either popularity field was missing.
             • Sets the IMDb ID if it was missing.
      5. If no IMDb data is retrieved and the description remains missing, it generates a description with an LLM.
      6. The function saves the current cached descriptions and scores to their respective CSV files every 100 movies.
      7. After processing all movies, it performs a final save and returns the updated descriptions DataFrame.

    Parameters:
      combined_dataframe (pd.DataFrame):
        DataFrame containing movie metadata (including columns such as 'movieId', 'title', and 'imdbId').
      model_name (str):
        Name of the language model used for generating descriptions.
      chat_format (str):
        Prompt format string used by the LLM to generate few-shot descriptions.
      descriptions_path (str):
        File path to the descriptions cache CSV file.
      scores_path (str):
        File path to the IMDb scores cache CSV file.
      url (str):
        LLM API endpoint URL.
      headers (dict):
        HTTP headers for API requests.
      start_movie_id (int, default=1):
        Movie ID from which to start processing (allows resuming from a specific point).
      max_retries (int, default=5):
        Maximum number of retry attempts for API calls (for both IMDb and LLM requests).
      delay_between_attempts (int, default=1):
        Delay in seconds between retry attempts.

    Returns:
      pd.DataFrame:
        A DataFrame containing all movie descriptions (updated cache).

    Notes:
      - Cached descriptions and scores are saved every 100 movies processed.
      - A progress message is printed every 10 seconds.
      - For IMDb scores, missing values are represented as None.
      - Both missing descriptions and missing score data are handled independently.
      - The function uses cached data to avoid repeated API calls; if data is missing,
        it attempts to retrieve it via the IMDb API using get_movie_with_retries() and falls back to LLM generation if necessary.
      - In this version, the check for popularity data now verifies that both raw and normalized popularity are present,
        and the cached IMDb ID is checked separately so that it can be updated if missing.
    """
    # Load cached data from CSV files.
    movie_scores_df = load_movie_scores(scores_path, combined_dataframe['movieId'].max())
    cached_descriptions = load_cached_descriptions(descriptions_path, combined_dataframe['movieId'].max())
    
    # Track progress using timestamps for periodic messages.
    last_print_time = time.time()

    # Process each row (movie) in the combined dataframe.
    for index, row in combined_dataframe.iterrows():
        current_time = time.time()
        if current_time - last_print_time >= 10:
            print(f"Processed {index+1} movies so far...")
            last_print_time = current_time

        movie_id = row['movieId']
        if movie_id < start_movie_id:
            continue
        
        # Retrieve the cached description.
        cached_description = cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'].iloc[0]
        
        # Retrieve cached IMDb rating, raw popularity, and normalized popularity.
        current_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0]
        current_raw_popularity = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'].iloc[0]
        current_normalized = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'].iloc[0]
        
        # Retrieve the cached IMDb ID.
        current_imdb_id = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'].iloc[0]
        
        # Check separately for missing IMDb rating, popularity, and IMDb ID.
        needs_rating = current_rating is None
        needs_popularity = (current_raw_popularity is None) or (current_normalized is None)
        needs_imdb_id = current_imdb_id is None

        # If description is missing or any of the rating/popularity/IMDb ID are missing, attempt retrieval.
        if cached_description == "" or needs_rating or needs_popularity or needs_imdb_id:
            movie_title = row['title']
            imdb_id = row['imdbId']
            
            # Attempt to fetch movie details with retries.
            movie, rating, popularity = get_movie_with_retries(
                imdb_id, movie_title, model_name, chat_format, url, headers, max_retries, delay_between_attempts
            )
            
            if movie:
                # Update description if missing.
                if cached_description == "":
                    if 'plot' in movie:
                        description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
                    else:
                        description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                    description = description.replace('\n', ' ').replace('\r', ' ').strip()
                    cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

                # Update IMDb ID if missing.
                if needs_imdb_id:
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'] = imdb_id

                # Update IMDb rating if missing.
                if needs_rating:
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'] = rating if rating is not None else None
                
                # Update popularity information if either raw or normalized popularity is missing.
                if needs_popularity:
                    normalized_popularity = normalize_popularity_score(popularity) if popularity else None
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'] = normalized_popularity
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'] = popularity if popularity is not None else None
            
            # If no IMDb data could be retrieved and the description remains missing, generate a description via LLM.
            elif cached_description == "":
                description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                description = description.replace('\n', ' ').replace('\r', ' ').strip()
                cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

        # Save cached data every 100 movies.
        if (index + 1) % 100 == 0:
            save_cached_descriptions(cached_descriptions, descriptions_path)
            save_movie_scores(movie_scores_df, scores_path)

    # Final save after processing all movies.
    save_cached_descriptions(cached_descriptions, descriptions_path)
    save_movie_scores(movie_scores_df, scores_path)

    return cached_descriptions


def generate_description_with_few_shot(
    movie_title: str,
    model_name: str,
    chat_format: str,
    url: str,
    headers: Dict[str, str]
) -> str:
    """
    Generate a movie description using few-shot prompting with a language model.

    This function constructs a prompt that includes a set of few-shot examples and a clear instruction,
    then sends it to a language model API via a POST request. The chat format for the "Phi-3-mini-4k-instruct-q4.gguf"
    model uses a single placeholder `{prompt}`, which is replaced with the concatenated few-shot examples
    and the movie title.

    Parameters:
      movie_title (str):
          The title of the movie for which to generate a description.
      model_name (str):
          The name of the language model to use (e.g., "Phi-3-mini-4k-instruct-q4.gguf").
      chat_format (str):
          A format string that structures the prompt. It should include a placeholder `{prompt}`
          where the concatenated few-shot examples and movie title will be inserted.
      url (str):
          The API endpoint URL to which the request is sent.
      headers (dict):
          A dictionary of HTTP headers (e.g., containing authorization) to include in the API request.

    Returns:
      str:
          The generated movie description as returned by the language model, with any leading/trailing
          whitespace removed. If the API call fails (i.e., the response status code is not 200), the function
          returns an empty string.

    Example:
        chat_format = (
            "<|user|>\n{prompt} <|end|>\n<|assistant|>"
        )
        description = generate_description_with_few_shot(
            movie_title="Interstellar",
            model_name="Phi-3-mini-4k-instruct-q4.gguf",
            chat_format=chat_format,
            url="https://api.example.com/v1/chat",
            headers={"Authorization": "Bearer YOUR_API_KEY"}
        )
    
    The resulting prompt sent to the API will consist of:
      - A system message that instructs the LLM to generate a concise movie description.
      - A user message formatted using `chat_format`, embedding few-shot examples and instructions.
    """
    # System instruction for the language model.
    role_instruction = (
        "You are a helpful assistant that generates concise movie descriptions. "
        "Do not use newlines in your response. The examples provided are for context only and should not appear in your output. "
        "Return only the description."
    )

    # Check if using phi-4 model and wrap system content accordingly
    if "phi-4-Q6_K" in model_name:
        wrapped_system = f"<|im_start|>system<|im_sep|>{role_instruction}<|im_end|>"
    else:
        wrapped_system = role_instruction

    # Few-shot examples defined within the function.
    few_shot_examples = (
        "Example 1:\n"
        "Movie title: Inception\n"
        "Description: A thief who steals corporate secrets through dream-sharing technology is tasked with implanting an idea into a CEO's mind.\n"
        "Example 2:\n"
        "Movie title: The Matrix\n"
        "Description: A computer hacker discovers his reality is an illusion and joins rebels to fight its controllers.\n"
    )
    
    # Construct the prompt by embedding few-shot examples and instructions into `chat_format`.
    prompt = chat_format.format(
        prompt=few_shot_examples + "\nGenerate a description for this movie:\n" + f"Movie title: {movie_title}\nDescription:"
    )

    # Create the payload for the API request.
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": wrapped_system},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 120,
        "temperature": 0
    }

    # Send POST request to language model API.
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        # Extract and return content from API response.
        return response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
    else:
        # Return an empty string if API call fails.
        return ""
    
def get_imdb_id_by_title(
    title: str,
    model_name: str,
    chat_format: str,
    url: str,
    headers: Dict[str, str],
    manual_selection: bool = False,
    results_limit: int = 20,
    page_limit: int = 5,
    fetch_full_details: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[float], Optional[int]]:
    """
    Retrieve the IMDb ID and additional movie details for a given movie title.

    This function uses the Cinemagoer library to search for a movie by its title. It supports two modes:
      - Manual selection: If `manual_selection` is True, the user is presented with a paginated list of results
        and prompted to select the correct movie.
      - Automatic matching: If `manual_selection` is False, the function attempts to find an exact title match.
        If no exact match is found, it can optionally use fuzzy matching via an external LLM (e.g., phi model).

    If `fetch_full_details` is True, the function retrieves additional information about the selected movie,
    such as director, cover URL, IMDb rating, and popularity (vote count).

    Parameters:
      title (str):
          The title of the movie to search for.
      model_name (str):
          The name of the language model to use for fuzzy matching (e.g., "Phi-3-mini-4k-instruct-q4.gguf").
      chat_format (str):
          A format string used to structure prompts for fuzzy matching with the language model.
      url (str):
          The API endpoint URL for sending requests to the language model.
      headers (Dict[str, str]):
          HTTP headers to include with API requests (e.g., authorization tokens).
      manual_selection (bool, default=False):
          Whether to prompt the user manually for selecting a movie from search results.
      results_limit (int, default=20):
          The maximum number of search results to retrieve from Cinemagoer.
      page_limit (int, default=5):
          The maximum number of search results displayed per page when manual selection is enabled.
      fetch_full_details (bool, default=False):
          Whether to fetch full movie details (e.g., director name, cover URL, IMDb rating) after selecting a movie.

    Returns:
      Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[float], Optional[int]]:
        A tuple containing:
          - imdb_id (str): The IMDb ID of the selected movie.
          - director (str): The name(s) of the director(s) if full details are fetched; otherwise None.
          - cover_url (str): The URL of the movie's cover image; defaults to None if unavailable.
          - imdb_title (str): The title of the movie as returned by IMDb.
          - imdb_rating (float): The IMDb rating if full details are fetched; otherwise 0.0.
          - popularity (int): The number of votes or popularity metric if full details are fetched; otherwise 0.

        If no match is found or an error occurs, all elements in the tuple will be None.

    Example:
        >>> imdb_id, director, cover_url, imdb_title, imdb_rating, popularity = get_imdb_id_by_title(
        ...     "Inception",
        ...     "Phi-3-mini-4k-instruct-q4.gguf",
        ...     "<|user|>\n{prompt} <|end|>\n<|assistant|>",
        ...     "https://api.example.com/v1/chat",
        ...     {"Authorization": "Bearer YOUR_API_KEY"},
        ...     manual_selection=False,
        ...     fetch_full_details=True
        ... )
        >>> print(imdb_id)  # Outputs something like "1375666"

    Notes:
      - When manual_selection is enabled, users can navigate through paginated search results and select a match interactively.
      - If no exact match is found and manual_selection is disabled, fuzzy matching can be employed using an external LLM.
      - In case of an error or if no match is found, an error message is displayed and all return values are set to None.
    """
    ia = Cinemagoer()
    try:

        # Search for the movie by title
        search_results = ia.search_movie(title, results_limit)

        # If we want manual selection, present the top few results and let the user pick
        if manual_selection:
            if not search_results:
                print(f"No search results found for '{title}'.")
                return None, None, None, None, None, None

            current_page = 0
            total_results = len(search_results)
            print(f"\nTotal Results: {total_results}")

            while True:
                start_index = current_page * page_limit
                end_index = min(start_index + page_limit, total_results)

                # Show the current page of search results
                print(f"Search results for '{title}' (Page {current_page + 1}):")
                for idx, movie in enumerate(search_results[start_index:end_index], start=start_index + 1):
                    year_info = f" ({movie.get('year')})" if 'year' in movie else ""
                    cover_url = movie.get('cover url', '')
                    print(f"{idx}. {movie['title']}{year_info}")
                    print(f"   Cover URL: {cover_url if cover_url else 'No cover image available.'}")

                # Ask the user which one they meant or if they want to see more results
                user_input = input(f"\nSelect the correct match by typing a number between {start_index + 1} and {end_index} (or 'n' for next page, 'p' for previous page, '0' to cancel): ").strip().lower()
                if user_input == '0':
                    print("Selection canceled.")
                    return None, None, None, None, None, None
                elif user_input == 'n':
                    if end_index < total_results:
                        current_page += 1
                    else:
                        print("You are on the last page. You can go back by entering 'p'.")
                elif user_input == 'p':
                    if current_page > 0:
                        current_page -= 1
                    else:
                        print("You are on the first page.")
                else:
                    try:
                        selection = int(user_input)
                        if start_index + 1 <= selection <= end_index:
                            selected_movie = search_results[selection - 1]
                            imdb_id = selected_movie.movieID
                            imdb_title = selected_movie['title']

                            # Fetch full details if requested
                            if fetch_full_details:
                                full_movie = ia.get_movie(imdb_id)
                                director = ', '.join([person['name'] for person in full_movie.get('director', [])])

                                # First try to use full-size cover, if that fails, use the standard size cover, otherwise, default to an empty string
                                cover_url = full_movie.get('full-size cover url', full_movie.get('cover url', ''))
                                imdb_rating = full_movie.get('rating', 0.0)
                                popularity = full_movie.get('votes', 0)

                            else:
                                director = ''
                                cover_url = selected_movie.get('cover url', '')
                                imdb_rating = 0.0
                                popularity = 0.0
                                

                            print(f"Selected '{imdb_title}' with IMDb ID {imdb_id}.")
                            return imdb_id, director, cover_url, imdb_title, imdb_rating, popularity
                        print(f"Please enter a valid number between {start_index + 1} and {end_index}, or 'n' for next page, 'p' for previous page, '0' to cancel.")
                    except ValueError:
                        print("Invalid input. Please enter a number or 'n' for next page, 'p' for previous page.")

        # If using not using manual select, check for exact match
        for movie in search_results:
            
            # Check for an exact title match with case sensitivity
            if movie['title'] == title:
                imdb_id = movie.movieID
                imdb_title = movie['title']

                # Fetch full details if requested
                if fetch_full_details:
                    full_movie = ia.get_movie(imdb_id)
                    director = ', '.join([person['name'] for person in full_movie.get('director', [])])
                    cover_url = full_movie.get('full-size cover url', full_movie.get('cover url', ''))
                    imdb_rating = full_movie.get('rating', 0.0)
                    popularity = full_movie.get('votes', 0.0)
                    
                else:
                    director = ''
                    cover_url = movie.get('cover url', '')
                    imdb_rating = 0.0
                    popularity = 0.0
                    
                # print(f"Found exact match for '{title}': IMDb ID is {imdb_id}")
                return imdb_id, director, cover_url, imdb_title, imdb_rating, popularity
                
        # If no exact match is found or manual selection is not used, use LLM to find the best match
        best_match = None
        highest_similarity = -1  

        role_instruction = ("You are a searching assistant that determines if a movie title matches what the user is looking for based on their input." 
                            "You must only respond with numbers between 0 and 1 under any and all circumstances without any text, formatting, or markdown. 1 is a sure match, and 0 is certainly not a match.")

        # Check if using phi-4 model and wrap system content accordingly
        if "phi-4-Q6_K" in model_name:
            wrapped_system = f"<|im_start|>system<|im_sep|>{role_instruction}<|im_end|>"
        else:
            wrapped_system = role_instruction

        # Few-shot examples for the prompt
        few_shot_examples = (
            "Example 1:\n"
            "User input: Avatar\n"
            "Movie title: Avatar: The Last Airbender\n"
            "How likely is it that this is the movie the user is looking for? (respond with a number between 0 and 1):\n"
            "0.5\n"
            "Example 2:\n"
            "User input: Star Wars\n"
            "Movie title: Star Wars: Episode IV - A New Hope\n"
            "How likely is it that this is the movie the user is looking for? (respond with a number between 0 and 1):\n"
            "0.9\n"
            "Example 3:\n"
            "User input: star wars\n"
            "Movie title: Robot Chicken: Star Wars\n"
            "How likely is it that this is the movie the user is looking for? (respond with a number between 0 and 1):\n"
            "0.2\n"
        )

        for movie in search_results: 

            # if movie.get('kind') == 'movie':
            movie_title = movie['title']

            prompt_content = (
                f"User input: {title}\n"
                f"Movie title: {movie_title}\n"
                "How likely is it that this is the movie the user is looking for? (respond with a number between 0 and 1):\n"
            )

            full_prompt = few_shot_examples + "Now, respond to the following prompt:\n" + prompt_content
            full_prompt_formatted = chat_format.format(prompt=full_prompt)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": wrapped_system},
                    {"role": "user", "content": full_prompt_formatted}
                ],
                "max_tokens": 5,
                "temperature": 0
            }
        
            # Make 3 attempts to get a valid similarity score for a search result
            max_score_attempts = 3
            similarity_score = -1

            for attempt in range(max_score_attempts):
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    content = response.json().get("choices", [])[0].get("message", {}).get("content", "0")
                    try:
                        # First try direct float conversion
                        similarity_score = float(content)
                        break  # Break if successful float conversion
                    except ValueError:
                        # If that fails, try to extract a number using regex
                        match = re.search(r'-?\d+\.?\d*', content)
                        if match:
                            similarity_score = float(match.group())
                            # Clamp the score between 0 and 1
                            similarity_score = max(0, min(1, similarity_score))
                            break  # Break if successful regex extraction
                        else:
                            print(f"Attempt {attempt + 1}: Could not extract similarity score for '{movie_title}' search result: {content}")
                            if attempt == max_score_attempts - 1:
                                print(f"All attempts failed for '{movie_title}' search result. Using default score for this result: -1")
                                similarity_score = -1  # Default low similarity score

            # Update best match if this movie has a higher similarity score
            if similarity_score > highest_similarity:
                highest_similarity = similarity_score
                best_match = movie       
            
        if best_match:
            imdb_id = best_match.movieID
            imdb_title = best_match['title']

            # Fetch full details if requested
            if fetch_full_details:
                full_movie = ia.get_movie(imdb_id)
                director = ', '.join([person['name'] for person in full_movie.get('director', [])])
                cover_url = full_movie.get('full-size cover url', full_movie.get('cover url', ''))
                imdb_rating = full_movie.get('rating', 0.0)
                popularity = full_movie.get('votes', 0.0)                
            else:
                director = ''
                cover_url = best_match.get('cover url', '')
                imdb_rating = 0.0
                popularity = 0.0
                
            return imdb_id, director, cover_url, imdb_title, imdb_rating, popularity
        else:
            print(f"No match found for title in IMDb movies: {title}")
            return None, None, None, None, None, None
    except IMDbError as e:
        print(f"An error occured while searching for movie {title}'s IMDb ID: {e}")
        return None, None, None, None, None, None

def get_movie_with_retries(imdb_id: str | int, movie_title: str, model_name: str, chat_format: str, url: str, headers: dict, max_retries: int = 10, delay: int = 1) -> Tuple[Optional[dict], Optional[float], Optional[int]]:
    """
    Attempts to fetch detailed movie or TV show data from IMDb, with support for partial data caching.
    The function makes multiple attempts to retrieve complete movie data, but will preserve the best partial
    data obtained across attempts if complete data cannot be fetched.

    Each attempt tries to gather:
    - Plot information from the movie object
    - IMDb rating
    - Vote count (as a measure of popularity)

    If any attempt fails with a 404 error (indicating an outdated/incorrect IMDb ID), the function will:
    1. Try to obtain an updated IMDb ID using get_imdb_id_by_title
    2. Retry data retrieval with the new identifier
    3. Continue this process until either complete data is retrieved or max retries reached

    The function tracks the best partial data seen across all attempts, so even if complete data
    cannot be retrieved, it returns the most complete information available.

    Parameters:
      imdb_id (str or int): The IMDb identifier for the movie
      movie_title (str): The title of the movie; used for IMDb ID lookups on 404 errors
      model_name (str): The language model name for fallback IMDb ID lookups
      chat_format (str): The prompt format string for the language model
      url (str): The API endpoint URL for LLM requests
      headers (dict): HTTP headers for API requests
      max_retries (int, optional): Maximum number of retry attempts (default: 10)
      delay (int, optional): Seconds to wait between retries (default: 1)

    Returns:
      tuple: A tuple (movie, rating, popularity) containing:
             - movie (dict): Best movie data object found, with plot if available
             - rating (float or None): Best IMDb rating found, or None if unavailable
             - popularity (int or None): Best vote count found, or None if unavailable

             Returns partial data if complete data cannot be retrieved within max_retries.
             Returns (None, None, None) only if no valid data could be retrieved.
    """
    ia = Cinemagoer()
    attempt = 0
    best_movie = None
    best_rating = None 
    best_popularity = None

    while attempt < max_retries:
        try:
            # Get the movie/show with all info
            movie = ia.get_movie(imdb_id, info=['main', 'plot'])
            
            if movie:
                # Update best data if current attempt has more info
                current_rating = movie.get('rating', None)
                current_popularity = movie.get('votes', None)
                has_plot = 'plot' in movie
                
                # Keep track of best data seen so far
                if has_plot and best_movie is None:
                    best_movie = movie
                if current_rating is not None:
                    best_rating = current_rating
                if current_popularity is not None:
                    best_popularity = current_popularity
                    
                # Return early if we have all data
                if has_plot and current_rating is not None and current_popularity is not None:
                    return movie, current_rating, current_popularity
                    
        except IMDbError as e:
            if 'HTTPError 404' in str(e):
                print(f"HTTP 404 error encountered for {movie_title}'s IMDb ID {imdb_id}. Attempting to find IMDb ID by title.")
                imdb_id, _, _, _, _, _ = get_imdb_id_by_title(movie_title, model_name, chat_format, url, headers)
                if not imdb_id:
                    print(f"Could not find the IMDb ID with the title '{movie_title}'.")
                    return best_movie, best_rating, best_popularity
            else:
                print(f"Attempt {attempt + 1} failed: {e}")

        attempt += 1
        if attempt < max_retries:
            time.sleep(delay)
        
    return best_movie, best_rating, best_popularity

def load_cached_descriptions(descriptions_path: str, max_movie_id: int) -> pd.DataFrame:
    """
    Loads and validates movie descriptions from a CSV file, ensuring continuous movie ID coverage.

    This function performs two main tasks:
    1. Loads existing descriptions if the CSV file exists
    2. Creates a new DataFrame with placeholder descriptions if the file doesn't exist

    The function guarantees that:
    - All movie IDs from 1 to max_movie_id are present
    - No gaps exist in the ID sequence
    - Missing descriptions are represented as empty strings
    - The returned DataFrame has exactly two columns: 'movieId' and 'description'

    Parameters:
        descriptions_path (str): 
            File path to the CSV containing movie descriptions.
            If the file exists, it must have at least 'movieId' and 'description' columns.
        
        max_movie_id (int): 
            The highest movie ID to include in the DataFrame.
            The function will create entries for all IDs from 1 to this number.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - movieId (int): Sequential IDs from 1 to max_movie_id
            - description (str): Movie descriptions, empty strings for missing data
            
            The DataFrame is guaranteed to:
            - Have exactly max_movie_id rows
            - Contain all IDs from 1 to max_movie_id
            - Have no null values (NaN are converted to empty strings)

    Example:
        >>> df = load_cached_descriptions("descriptions.csv", 3)
        >>> print(df)
           movieId description
        0       1           
        1       2 A great movie
        2       3           
    """
    if os.path.exists(descriptions_path):
        cached_descriptions = pd.read_csv(descriptions_path)
        # Ensure all movie IDs from 1 to max_movie_id are present
        all_movie_ids = pd.DataFrame({'movieId': range(1, max_movie_id + 1)})
        complete_cached_descriptions = pd.merge(all_movie_ids, cached_descriptions, on='movieId', how='left')
        complete_cached_descriptions['description'] = complete_cached_descriptions['description'].fillna("")
    else:
        # Create a DataFrame with all movie IDs from 1 to max_movie_id
        complete_cached_descriptions = pd.DataFrame({
            'movieId': range(1, max_movie_id + 1), 
            'description': [""] * max_movie_id
        })

    return complete_cached_descriptions
    
def save_cached_descriptions(cached_descriptions: pd.DataFrame, descriptions_path: str) -> None:
    """
    Saves movie descriptions to a CSV file with retry mechanism for handling file access conflicts.

    This function attempts to save the movie descriptions DataFrame to a CSV file, implementing:
    1. Directory creation if it doesn't exist
    2. Multiple retry attempts if the file is locked/in use
    3. User prompts to resolve file access conflicts
    4. UTF-8 encoding to support special characters
    
    Parameters:
        cached_descriptions (pd.DataFrame): 
            DataFrame containing movie descriptions with required columns:
            - movieId (int): Unique identifier for each movie
            - description (str): Text description of the movie's plot/content
            The DataFrame should not contain duplicate movieIds.
            
        descriptions_path (str): 
            Full file path where the CSV will be saved.
            If the directory structure doesn't exist, it will be created.

    Returns:
        None
            Prints "File saved successfully." on success
            Prints error messages and retry prompts on failure

    Raises:
        No exceptions are raised; all errors are handled internally with retries

    Notes:
        - Makes up to 10 retry attempts if file saving fails
        - On each retry (except the last), prompts user to close conflicting applications
        - Uses UTF-8 encoding to support international characters in descriptions
        - Does not save the DataFrame index to the CSV file
        - Creates any missing directories in the path automatically

    Example:
        >>> descriptions_df = pd.DataFrame({
        ...     'movieId': [1, 2],
        ...     'description': ['A great movie', 'Another great movie']
        ... })
        >>> save_cached_descriptions(descriptions_df, 'data/cached_descriptions.csv')
        File saved successfully.
    """
    max_retries = 10

    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(descriptions_path), exist_ok=True)
            cached_descriptions.to_csv(descriptions_path, index=False, encoding='utf-8')
            print("File saved successfully.")
            break
        except (IOError, OSError) as e:
            print(f"Attempt {attempt + 1}: Error saving descriptions: {e}")
            if attempt < max_retries - 1:
                print("Please close any applications using the file and press Enter to retry...")
                input()
            else:
                print("Failed to save the file after multiple attempts. Please ensure the file is not open in another application.")

def get_movie_descriptions(
    top_n_movies: List[Tuple[Any, str, float]], 
    combined_dataframe: pd.DataFrame,
    model_name: str,
    chat_format: str,
    descriptions_path: str, 
    scores_path: str,
    url: str,
    headers: Dict[str, str],
    max_retries: int,
    delay_between_attempts: int
) -> Dict[Any, str]:
    """
    Retrieves and caches movie descriptions and IMDb metadata for a list of recommended movies.

    This function processes each movie in the top_n_movies list by:
    1. Checking if description exists in cache
    2. Checking if IMDb rating and popularity scores exist in cache
    3. For missing data:
       - Attempts to fetch from IMDb API
       - Falls back to LLM-generated descriptions if IMDb fetch fails
       - Updates caches with any new data retrieved
    4. Returns a dictionary mapping movie IDs to their descriptions

    Parameters:
        top_n_movies (List[Tuple[Any, str, float]]): 
            List of tuples containing (movie_id, movie_title, estimated_rating) for recommended movies
        combined_dataframe (pd.DataFrame):
            DataFrame containing movie metadata with at least movieId and imdbId columns
        model_name (str):
            Name of language model for generating descriptions when IMDb fails
        chat_format (str): 
            Prompt template string for LLM description generation
        descriptions_path (str):
            File path to descriptions cache CSV file
        scores_path (str):
            File path to IMDb scores cache CSV file
        url (str):
            LLM API endpoint URL for description generation
        headers (Dict[str, str]):
            HTTP headers for LLM API requests
        max_retries (int):
            Maximum number of retry attempts for IMDb API calls
        delay_between_attempts (int):
            Seconds to wait between retry attempts

    Returns:
        Dict[Any, str]:
            Dictionary mapping movie IDs to their descriptions, using cached descriptions
            when available and generating new ones as needed

    Notes:
        - Caches are loaded at start and saved after processing all movies
        - Shows progress message every 10 seconds during processing
        - Updates both description and score caches with new data
        - Uses IMDb API first, falls back to LLM generation if needed
        - Handles IMDb API errors with configurable retries
        - Ensures descriptions are single-line by removing newlines
    """
    # Initialize an empty dictionary to store movie descriptions
    descriptions = {}

    # Load cached data
    cached_descriptions = load_cached_descriptions(descriptions_path, combined_dataframe['movieId'].max())
    movie_scores_df = load_movie_scores(scores_path, combined_dataframe['movieId'].max())

    # Track the last time the message was printed
    last_print_time = time.time()

    # Iterate over the list of top N movies
    for movie_id, movie_title, _ in top_n_movies:
        
        current_time = time.time()
        # Check if 10 seconds have passed since the last message print
        if current_time - last_print_time >= 10:
            print("Retrieving movie data...")
            last_print_time = current_time

        # Check if we have a cached description
        cached_description = cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'].iloc[0]
        
        # Get the current values in movie scores df for the movie
        current_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0]
        current_raw = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'].iloc[0]
        current_normalized = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'].iloc[0]
        current_imdb_id = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'].iloc[0]

        # Get the IMDb ID for the current movie in combined df
        imdb_id = combined_dataframe.loc[combined_dataframe['movieId'] == movie_id, 'imdbId'].values[0]
        
        # If we need to fetch either description or scores, make the IMDb API call
        if cached_description == "" or current_rating is None or current_raw is None or current_normalized is None or current_imdb_id is None:
            movie, rating, votes = get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, url, headers, max_retries, delay_between_attempts)
            
            if movie:
                # Get description if needed
                if cached_description == "":
                    if 'plot' in movie:
                        description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
                    else:
                        # Could not find a description, as a last resort, generate a description using few shot prompting
                        description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                        
                    # Ensure the description is a single line
                    description = description.replace('\n', ' ').replace('\r', ' ').strip()
                    
                    # Update description cache
                    cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description
                else:
                    description = cached_description

                # Make sure IMDB ID is the same in both dataframes
                if current_imdb_id is None:
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'] = imdb_id  

                # Update scores if rating is None
                if current_rating is None and rating is not None:
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'] = rating

                # Update popularity information if either popularity measure is missing
                if (current_raw is None or current_normalized is None) and votes is not None:
                    normalized_popularity = normalize_popularity_score(votes)
                    if current_raw is None:
                        movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'] = votes
                    if current_normalized is None:
                        movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'] = normalized_popularity
            else:
                # If movie data fetch failed but we need a description
                if cached_description == "":
                    description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                    description = description.replace('\n', ' ').replace('\r', ' ').strip()
                    cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description
                else:
                    description = cached_description
        else:
            description = cached_description

        # Store the description in the return dictionary
        descriptions[movie_id] = description

    # Save updated caches
    save_cached_descriptions(cached_descriptions, descriptions_path)
    save_movie_scores(movie_scores_df, scores_path)

    return descriptions

def find_top_n_similar_movies(
    user_input: str,
    movie_descriptions: Dict[Any, str],
    id_to_title: Dict[Any, str],
    model_name: str,
    chat_format: str,
    n: int,
    url: str,
    headers: Dict[str, str],
    movie_scores_df: Optional[pd.DataFrame] = None,
    prioritize_ratings: bool = False,
    prioritize_popular: bool = False,
    favorite_movies: Optional[List[str]] = None,
    num_favorites: int = 3,
    max_retries: int = 3,
    date_range: Optional[Tuple[int, int]] = None
) -> List[Tuple[Any, float]]:
    """
    Uses a Large Language Model (LLM) to find movies most similar to a user's preferences.

    This function evaluates each candidate movie by:
    1. Constructing a detailed prompt incorporating:
       - User's stated preferences
       - User's favorite movies (if provided)
       - Preferred release date range
       - Rating preferences
       - Popularity preferences
    2. Using few-shot examples to guide the LLM in generating similarity scores
    3. Making API calls to the LLM to evaluate each movie
    4. Handling response parsing and retry logic
    5. Sorting and returning the top N most similar movies

    Parameters:
        user_input (str):
            Natural language description of user's movie preferences.

        movie_descriptions (Dict[Any, str]):
            Maps movie IDs to their plot descriptions.

        id_to_title (Dict[Any, str]):
            Maps movie IDs to their titles.

        model_name (str):
            Name of LLM to use (e.g., "phi-4-Q6_K").
            Affects system message formatting.

        chat_format (str):
            Template for structuring LLM prompts.
            Must contain {prompt} placeholder.

        n (int):
            Number of top movies to return.

        url (str):
            LLM API endpoint URL.

        headers (Dict[str, str]):
            HTTP headers for API requests.

        movie_scores_df (Optional[pd.DataFrame]):
            DataFrame with columns:
            - movieId: Movie identifier
            - imdb_rating: Rating on 0-10 scale
            - normalized_popularity: Score on 0-100 scale

        prioritize_ratings (bool, default=False):
            Whether to emphasize highly-rated movies.

        prioritize_popular (bool, default=False):
            Whether to emphasize popular movies.

        favorite_movies (Optional[List[str]], default=None):
            List of user's favorite movie titles.

        num_favorites (int, default=3):
            Maximum number of favorite movies to include in prompt.

        max_retries (int, default=3):
            Maximum API call attempts per movie.

        date_range (Optional[Tuple[int, int]], default=None):
            (start_year, end_year) for preferred movies.

    Returns:
        List[Tuple[Any, float]]:
            List of (movie_id, similarity_score) tuples.
            Sorted by similarity score descending.
            Length = min(n, len(movie_descriptions))
            Scores are clamped to [-1.0, 1.0].

    Prompt Structure:
        The function constructs prompts with:
        1. System instruction defining the task
        2. Few-shot examples showing desired behavior
        3. User preferences and constraints
        4. Movie details to evaluate:
           - Title
           - IMDb rating (if available)
           - Popularity score (if available)
           - Plot description

    Example:
        >>> movies_description = {
        ...     260: "A long time ago in a galaxy far, far away...",
        ...     45517: "Lightning McQueen, a hotshot rookie..."
        ... }
        >>> titles = {260: "Star Wars: Episode IV - A New Hope (1977)", 45517: "Cars (2006)"}
        >>> scores = pd.DataFrame({
        ...     'movieId': [260, 45517],
        ...     'imdb_rating': [8.6, 7.3],
        ...     'normalized_popularity': [100, 49]
        ... })
        >>> preferences = "I enjoy movies with epic adventures, heartwarming stories, and themes of friendship and personal growth."
        >>> similar = find_top_n_similar_movies(
        ...     preferences, movies, titles, "model", "{prompt}",
        ...     1, "api.example.com", {"auth": "key"},
        ...     scores, favorite_movies=["Toy Story (1995)", "The Lion King (1994)", "Finding Nemo (2003)"],
        ... )
        >>> print(similar)
        [(260, 0.9), (45517, 0.7)]

    Notes:
        - Uses temperature=0.7 for API calls to balance consistency and creativity
        - Implements regex fallback for parsing non-numeric LLM responses
        - Caches similarity scores to avoid redundant API calls
        - Prints error messages for failed API calls or parsing issues
        - Supports both standard and Phi-4 model prompt formats
    """
    similarity_scores = []

    # Define the role and instructions with rating scale explanation
    role_instruction = (
        "You are a movie recommendation assistant. "
        "Your task is to evaluate how well a movie description aligns with a user's stated preferences and their favorite movies."
        "Always respond with a number between -1.0 and 1.0, where:\n"
        "-1.0 means the movie goes completely against their preferences,\n"
        "0 means neutral or there isn't enough information,\n"
        "1.0 is a perfect match. You must respond with only the number, without any additional text or formatting under all circumstances.\n"
    )

    # Check if using phi-4 model and wrap system content accordingly
    if "phi-4-Q6_K" in model_name:
        wrapped_system = f"<|im_start|>system<|im_sep|>{role_instruction}<|im_end|>"
    else:
        wrapped_system = role_instruction

    # Add example input and output to guide the LLM
    few_shot_examples = (
        "Example 1:\n"
        "User input: I love science fiction with deep philosophical themes.\n"
        "User's favorite movies:\n"
        "Movie title: The Matrix (1999)\n"
        "Movie title: Blade Runner (1982)\n"
        "Movie title: Interstellar (2014)\n"
        "Preferred Release Date Range: I prefer movies released between 1980 and 2020.\n"
        "New movie to evaluate:\n"
        "Movie title: Inception (2010)\n"
        "Movie description: A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the project and his team to disaster.\n"
        "Rate how likely you think the movie aligns with the user's interests (respond with a number in range [-1, 1]):\n"
        "0.9\n"
        "Example 2:\n"
        "User input: I enjoy light-hearted comedies with a lot of humor.\n"
        "User's favorite movies:\n"
        "Movie title: The Hangover (2009)\n"
        "Movie title: Superbad (2007)\n"
        "Movie title: Step Brothers (2008)\n"
        "Preferred Release Date Range: I prefer movies released between 2000 and 2010.\n"
        "New movie to evaluate:\n"
        "Movie title: The Dark Knight (2008)\n"
        "Movie description: Set within a year after the events of Batman Begins (2005), Batman, Lieutenant James Gordon, and new District Attorney Harvey Dent successfully begin to round up the criminals that plague Gotham City, until a mysterious and sadistic criminal mastermind known only as \"The Joker\" appears in Gotham, creating a new wave of chaos. Batman's struggle against The Joker becomes deeply personal, forcing him to \"confront everything he believes\" and improve his technology to stop him. A love triangle develops between Bruce Wayne, Dent, and Rachel Dawes.\n"
        "Rate how likely you think the movie aligns with the user's interests (respond with a number in range [-1, 1]):\n"
        "-0.7\n"
        "Example 3:\n"
        "User input: I am fascinated by historical documentaries.\n"
        "User's favorite movies:\n"
        "Movie title: They shall not grow old (2018)\n"
        "Movie title: Apollo 11 (2019)\n"
        "Movie title: 13th (2016)\n"
        "Preferred Release Date Range: I prefer movies released between 2010 and 2020.\n"
        "New movie to evaluate:\n"
        "Movie title: The Lord of the Rings: The Fellowship of the Ring (2001)\n"
        "Movie description: A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.\n"
        "Rate how likely you think the movie aligns with the user's interests (respond with a number in range [-1, 1]):\n"
        "-0.5\n"
    )

    # Add favorite movies to the prompt if provided
    favorite_movies_prompt= ""
    if favorite_movies:
        num_to_include = min(num_favorites, len(favorite_movies))
        favorite_movies_prompt="""User's favorite movies:\n"""
        for title in favorite_movies[:num_to_include]:
            favorite_movies_prompt += f"Movie title: {title}\n"

    # Add date range to the user input if provided
    date_range_prompt = ""
    if date_range:
        date_range_prompt = f"Preferred Release Date Range: I prefer movies released between {date_range[0]} and {date_range[1]}."

    for movie_id, description in movie_descriptions.items():

        # Get the title for the prompt
        movie_title = id_to_title[movie_id]

        # Get movie's rating and popularity from movie_scores_df
        imdb_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0] if movie_scores_df is not None else 0
        popularity = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'].iloc[0] if movie_scores_df is not None else 0

        # Build the prompt content with conditional rating/popularity lines
        prompt_content = (
            f"User input: {user_input} "
            f"{'I prefer movies with high IMDb ratings. ' if prioritize_ratings else ''}"
            f"{'I prefer popular/trending movies.' if prioritize_popular else ''}\n"
            f"{favorite_movies_prompt}\n"
            f"{date_range_prompt}\n"
            "New movie to evaluate:\n"
            f"Movie title: {movie_title}\n"
        )

        # Only add rating line if it exists and is not None
        if imdb_rating is not None:
            prompt_content += f"IMDb Rating: {imdb_rating}/10\n"

        # Only add popularity line if it exists and is not None
        if popularity is not None:
            prompt_content += f"Popularity Score: {popularity}/100\n"

        prompt_content += (
            f"Movie description: {description}\n"
            "Rate how likely you think the movie aligns with the user's interests (respond with a number in range [-1, 1]):\n"
        )

        full_prompt = few_shot_examples + "Now, respond to the following prompt:\n" + prompt_content

        # Format the prompt using the provided chat format
        full_prompt = chat_format.format(prompt=full_prompt)

        payload = {
            "model": model_name,  # Use passed model name
            "messages": [
                {"role": "system", "content": wrapped_system},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 5,
            "temperature": 0.7  # Temperature of 0.7 to encourage adaptability and prevent the LLM from repeating mistakes
        }

        similarity_score = 0.0  # Default score
        retries = 0

        while retries < max_retries:
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                content = response.json().get("choices", [])[0].get("message", {}).get("content", "0")
                try:
                    similarity_score = float(content)
                    # Clamp the similarity score between -1 and 1
                    if similarity_score < -1.0:
                        similarity_score = -1.0
                    elif similarity_score > 1.0:
                        similarity_score = 1.0
                    break  # Exit the retry loop if successful
                except ValueError:
                    # Attempt to extract a number from the response string
                    match = re.search(r"-?\d+(\.\d+)?", content)
                    if match:
                        similarity_score = float(match.group())
                        # Clamp the similarity score between -1 and 1
                        if similarity_score < -1.0:
                            similarity_score = -1.0
                        elif similarity_score > 1.0:
                            similarity_score = 1.0
                        break  # Exit the retry loop if successful
                    else:
                        print(f"Could not convert similarity score to float for movie '{movie_title}': {content}")
            else:
                print(f"Request failed for movie '{movie_title}' with status code {response.status_code}")

            retries += 1

        # Debug: Print each similarity score and its type
        # print(f"Movie Title: {movie_title}, Movie ID: {movie_id}, Similarity Score: {similarity_score}, Type: {type(similarity_score)}")

        similarity_scores.append((movie_id, similarity_score))

        # Print the role instruction, full prompt, and response if the similarity score is above 0.5
        '''
        if similarity_score > 0.5:
            print(f"Role Instruction:\n{role_instruction}\n")
            print(f"Full Prompt for movie '{movie_title}':\n{full_prompt}\n")
            print(f"Response for movie '{movie_title}': {content}\n")
        '''

    # Sort the movies by similarity score in descending order and select the top N
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_n_movies = similarity_scores[:n]
 
    return top_n_movies

def get_user_favorite_movies(
    user_id: Any,
    ratings_dataframe: pd.DataFrame,
    id_to_title: Dict[Any, str],
    num_favorites: int = 3
) -> List[str]:
    """
    Retrieves a user's favorite movie titles based on their ratings.

    This function identifies the top N (`num_favorites`) movies rated by a given user
    in the provided `ratings_dataframe` and returns a list of their titles
    using the `id_to_title` mapping.

    Parameters:
        user_id (Any):
            The ID of the user.
        ratings_dataframe (pd.DataFrame):
            The DataFrame containing user ratings with at least columns 'userId', 'movieId', and 'rating'.
        id_to_title (Dict[Any, str]):
            A dictionary mapping movie IDs to their titles.
        num_favorites (int, optional):
            The number of favorite movies to retrieve. Defaults to 3.

    Returns:
        List[str]:
            A list of the user's favorite movie titles.
    """
    user_ratings = ratings_dataframe[ratings_dataframe['userId'] == user_id]
    favorite_movies = user_ratings.nlargest(num_favorites, 'rating')['movieId'].tolist()
    favorite_movie_titles = [id_to_title[movie_id] for movie_id in favorite_movies]
    return favorite_movie_titles
    
def generate_preferences_from_rated_movies(
    rated_movies: List[Tuple[Any, str, str, float]],
    model_name: str,
    chat_format: str,
    url: str,
    headers: Dict[str, str]
) -> Optional[str]:
    """
    Generate a user preference summary based on their rated movies using few-shot LLM prompting.
    
    This function takes a user's rated movies and generates a natural language summary of their 
    movie preferences. It uses few-shot prompting with example movie ratings and preference 
    summaries to guide the language model in generating a personalized preference summary.

    Parameters:
        rated_movies (List[Tuple[Any, str, str, float]]): 
            List of tuples containing rated movie information:
            - movie_id (Any): Unique identifier for the movie
            - title (str): Movie title
            - description (str): Movie plot/content description
            - rating (float): User's rating for the movie (typically 0.5-5.0)
        
        model_name (str):
            Name of the language model to use for generating the summary.
            Supports special formatting for phi-4 model variants.
        
        chat_format (str):
            Template string for structuring the prompt to the language model.
            Should contain a {prompt} placeholder for the actual content.
        
        url (str):
            API endpoint URL for the language model service.
        
        headers (Dict[str, str]):
            HTTP headers for the API request, typically including authorization
            and content type headers.

    Returns:
        Optional[str]: 
            A single-line string summarizing the user's movie preferences based on 
            their rated movies, written in first person.
            Returns None if the API request fails.

    Example:
        >>> movies = [
        ...     (1, "The Matrix", "A computer hacker discovers...", 5.0),
        ...     (2, "Inception", "A thief who steals corporate...", 4.5)
        ... ]
        >>> preferences = generate_preferences_from_rated_movies(
        ...     movies, "gpt-4o-mini", "{prompt}", "https://api.example.com", 
        ...     {"Authorization": "Bearer xyz"}
        ... )
        >>> print(preferences)
        "I enjoy science fiction movies with mind-bending plots and philosophical themes."

    Notes:
        - Uses max_tokens=60 to keep summaries concise
        - Removes all newlines from the generated summary
        - Uses temperature=0 for consistent outputs
        - Handles HTTP errors and returns None on failure
    """

    max_tokens = 60

    role_instruction = (
        "You are a helpful assistant that generates comprehensive user preference summaries based on multiple movies that a user has rated. "
        "Ensure the preferences are written in the first person without newlines. "
        "The instructions and examples exist to help you understand the context and how to format your response, do not include them in your response. "
        "You should match the format of the user preferences responses in the examples exactly with nothing else in your response."
    )

    # Check if using phi-4 model and wrap system content accordingly
    if "phi-4-Q6_K" in model_name:
        wrapped_system = f"<|im_start|>system<|im_sep|>{role_instruction}<|im_end|>"
    else:
        wrapped_system = role_instruction

    # Few-shot examples with real movies and aggregated preferences
    few_shot_examples = (
        "Example 1:\n"
        "Movies:\n"
        "1. Movie title: The Shawshank Redemption\n"
        "   Description: A banker convicted of uxoricide forms a friendship over a quarter century with a hardened convict, while maintaining his innocence and trying to remain hopeful through simple compassion.\n"
        "   User rating: 5.0\n"
        "2. Movie title: The Godfather\n"
        "   Description: The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.\n"
        "   User rating: 4.5\n"
        "User preferences: I enjoy movies with deep character development, themes of redemption, and complex family dynamics.\n"
        "Example 2:\n"
        "Movies:\n"
        "1. Movie title: The Matrix\n"
        "   Description: A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.\n"
        "   User rating: 4.5\n"
        "2. Movie title: Inception\n"
        "   Description: A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the project and his team to disaster.\n"
        "   User rating: 4.0\n"
        "User preferences: I love movies with mind-bending plots, action-packed sequences, and philosophical themes.\n"
    )

    # Combine rated movie descriptions with titles and ratings
    combined_descriptions = "\n".join(
        f"Movie title: {title}\nDescription: {description}\nUser rating: {rating}"
        for _, title, description, rating in rated_movies
    )

    # Format the prompt using the provided chat format
    prompt = chat_format.format(
        prompt=few_shot_examples + 
        f"Now, based on the following movie titles, descriptions, and ratings, using {max_tokens} tokens or less, "
        "generate a comprehensive user preference summary in the first person, without newlines:\n" + 
        combined_descriptions
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": wrapped_system},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        summary = response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()

        # Post-process to ensure the summary is a single line
        summary = summary.replace('\n', ' ').replace('\r', ' ').strip()

        return summary
    except requests.exceptions.RequestException as e:
        print("An error occurred while making the request:", e)
        return None

def load_user_preferences(preferences_path: str, max_user_id: int) -> pd.DataFrame:
    """
    Load and validate user movie preferences from a CSV file, ensuring continuous user ID coverage.

    This function handles loading existing preferences or creating a new preferences DataFrame
    that covers all user IDs from 1 to max_user_id. It ensures backwards compatibility and
    proper initialization of all preference fields.

    Parameters:
        preferences_path (str):
            Path to the CSV file containing user preferences.
            If the file exists, it must contain at least a 'userId' column.
            Additional columns will be initialized if missing.
        
        max_user_id (int):
            The highest user ID to include in the DataFrame.
            The function ensures entries exist for all IDs from 1 to this number.

    Returns:
        pd.DataFrame:
            A DataFrame containing user preferences with the following columns:
            - userId (int): Sequential user IDs from 1 to max_user_id
            - preferences (str): Text description of user's movie preferences (empty string if none)
            - date_range (Optional[Tuple[int, int]]): User's preferred movie release date range
              in format (start_year, end_year), or None if not specified
            - prioritize_ratings (Optional[bool]): Whether user prefers highly rated movies
            - prioritize_popular (Optional[bool]): Whether user prefers popular/trending movies

    Notes:
        - If the CSV file exists:
            * Loads existing preferences
            * Ensures all user IDs from 1 to max_user_id are present
            * Fills missing preferences with empty strings
            * Initializes missing columns with None values
            * Converts empty date_range strings to None
            * Handles NaN values in boolean columns
        
        - If the CSV file doesn't exist:
            * Creates a new DataFrame with all required columns
            * Initializes preferences with empty strings
            * Initializes other columns with None values

        - The returned DataFrame is guaranteed to:
            * Have exactly max_user_id rows
            * Contain all user IDs from 1 to max_user_id
            * Have properly typed values in all columns
            * Have no NaN values (converted to empty strings or None)

    Example:
        >>> df = load_user_preferences("preferences.csv", 3)
        >>> print(df)
           userId preferences date_range  prioritize_ratings  prioritize_popular
        0       1                  None              None              None
        1       2    I like...    (1990, 2020)      True             False
        2       3                  None              None              None
    """
    if os.path.exists(preferences_path):
        # Load existing preferences
        preferences_df = pd.read_csv(preferences_path)
        
        # Create a DataFrame with all user IDs
        all_user_ids = pd.DataFrame({'userId': range(1, max_user_id + 1)})
        
        # Merge existing preferences with complete user ID range
        complete_preferences_df = pd.merge(all_user_ids, preferences_df, on='userId', how='left')
        
        # Fill missing preferences with empty strings
        complete_preferences_df['preferences'] = complete_preferences_df['preferences'].fillna("")
        
        # Handle date_range column
        if 'date_range' in complete_preferences_df.columns:
            complete_preferences_df['date_range'] = complete_preferences_df['date_range'].fillna(value="")
            complete_preferences_df['date_range'] = complete_preferences_df['date_range'].apply(
                lambda x: None if x == "" else x
            )
        else:
            complete_preferences_df['date_range'] = None
        
        # Handle prioritize_ratings column
        if 'prioritize_ratings' in complete_preferences_df.columns:
            complete_preferences_df['prioritize_ratings'] = complete_preferences_df['prioritize_ratings'].apply(
                lambda x: None if pd.isna(x) else x
            )
        else:
            complete_preferences_df['prioritize_ratings'] = None
        
        # Handle prioritize_popular column    
        if 'prioritize_popular' in complete_preferences_df.columns:
            complete_preferences_df['prioritize_popular'] = complete_preferences_df['prioritize_popular'].apply(
                lambda x: None if pd.isna(x) else x
            )
        else:
            complete_preferences_df['prioritize_popular'] = None
    else:
        # Create new DataFrame with all required columns
        complete_preferences_df = pd.DataFrame({
            'userId': range(1, max_user_id + 1),
            'preferences': [""] * max_user_id,
            'date_range': [None] * max_user_id,
            'prioritize_ratings': [None] * max_user_id,
            'prioritize_popular': [None] * max_user_id
        })

    return complete_preferences_df

def save_user_preferences(preferences_df: pd.DataFrame, preferences_path: str) -> None:
    """
    Save user movie preferences to a CSV file with retry mechanism for handling file access conflicts.

    This function attempts to save the preferences DataFrame to disk, implementing:
    1. Directory creation if it doesn't exist
    2. Multiple retry attempts if the file is locked/in use
    3. UTF-8 encoding to preserve special characters
    4. User prompts to resolve file access issues

    Parameters:
        preferences_df (pd.DataFrame):
            DataFrame containing user preferences with required columns:
            - userId (int): Unique identifier for each user
            - preferences (str): Text description of user's movie preferences
            - date_range (Optional[Tuple[int, int]]): User's preferred movie release years
            - prioritize_ratings (Optional[bool]): Whether user prefers highly rated movies
            - prioritize_popular (Optional[bool]): Whether user prefers popular movies
            The DataFrame should not contain duplicate userIds.
        
        preferences_path (str):
            Full file path where the CSV will be saved.
            If the directory structure doesn't exist, it will be created.

    Returns:
        None
            Prints success message on successful save
            Prints error messages and retry prompts if save fails

    Raises:
        No exceptions are raised; all errors are handled internally

    Notes:
        - Makes up to 10 retry attempts if file saving fails
        - Prompts user to close conflicting applications between retries
        - Uses UTF-8 encoding to support international characters
        - Does not save the DataFrame index to the CSV file
        - Creates any missing directories in the path automatically

    Example:
        >>> preferences_df = pd.DataFrame({
        ...     'userId': [1, 2],
        ...     'preferences': ['I enjoy sci-fi', 'I prefer comedies'],
        ...     'date_range': [(1990, 2020), None],
        ...     'prioritize_ratings': [True, False],
        ...     'prioritize_popular': [False, True]
        ... })
        >>> base_path = os.path.join('Datasets', 'Movie_Lens_Datasets', 'ml-latest-small')
        >>> preferences_path = os.path.join(base_path, 'preferences.csv')
        >>> save_user_preferences(preferences_df, preferences_path)
        Preferences saved successfully.
    """
    max_retries = 10

    for attempt in range(max_retries):
        try:
            # Create directory structure if it doesn't exist
            os.makedirs(os.path.dirname(preferences_path), exist_ok=True)
            
            # Save DataFrame to CSV without index
            preferences_df.to_csv(preferences_path, index=False, encoding='utf-8')
            print("Preferences saved successfully.")
            break
        
        except (IOError, OSError) as e:
            print(f"Attempt {attempt + 1}: Error saving preferences: {e}")
            
            # If not the last attempt, prompt for retry
            if attempt < max_retries - 1:
                print("Please close any applications using the file and press Enter to retry...")
                input()
            else:
                print("Failed to save the file after multiple attempts. Please ensure the file is not open in another application.")

def retrieve_all_descriptions(
    combined_dataframe: pd.DataFrame,
    model_name: str,
    chat_format: str,
    descriptions_path: str,
    scores_path: str,
    url: str,
    headers: Dict[str, str],
    start_movie_id: int = 1,
    max_retries: int = 5,
    delay_between_attempts: int = 1
) -> pd.DataFrame:
    """
    Retrieves and caches movie descriptions and IMDb metadata for all movies in the dataset.

    This function processes each movie in the combined_dataframe by performing the following steps:
    1. Loading existing cached scores and descriptions from local CSV files
    2. For each movie with movieId >= start_movie_id, checking whether:
       - The description is missing (empty string)
       - The IMDb ID is missing (None)
       - The IMDb rating is missing (None)
       - Either the raw popularity OR normalized popularity score is missing (None)
    3. For movies with any missing data:
       a. Attempts to fetch movie details from IMDb API using get_movie_with_retries()
       b. If successful:
          - Updates description if missing by:
            * Using IMDb plot if available
            * Falling back to LLM generation if plot missing
          - Updates IMDb ID if missing
          - Updates IMDb rating if missing
          - Updates both raw and normalized popularity if either is missing
       c. If IMDb fetch fails but description is missing:
          - Generates description using LLM as fallback
    4. Saves updated cache files:
       - Every 100 movies processed
       - After processing all movies

    Parameters:
        combined_dataframe (pd.DataFrame):
            DataFrame containing movie metadata with required columns:
            - movieId (int): Unique identifier for each movie
            - title (str): Movie title
            - imdbId (str): IMDb identifier

        model_name (str):
            Name of the language model used for generating descriptions
            when IMDb data cannot be retrieved

        chat_format (str):
            Template string for formatting prompts to the language model

        descriptions_path (str):
            File path to CSV containing cached movie descriptions

        scores_path (str):
            File path to CSV containing cached IMDb metadata:
            - IMDb IDs
            - IMDb ratings (0-10 scale)
            - Raw popularity scores
            - Normalized popularity scores (0-100 scale)

        url (str):
            API endpoint URL for the language model service

        headers (Dict[str, str]):
            HTTP headers for API requests (e.g., authorization tokens)

        start_movie_id (int, optional):
            Movie ID from which to start processing. Defaults to 1.
            Useful for resuming interrupted processing.

        max_retries (int, optional):
            Maximum number of retry attempts for IMDb API calls.
            Defaults to 5.

        delay_between_attempts (int, optional):
            Delay in seconds between retry attempts.
            Defaults to 1.

    Returns:
        pd.DataFrame:
            DataFrame containing all movie descriptions with columns:
            - movieId (int): Unique identifier for each movie
            - description (str): Movie plot description (from IMDb or LLM)

    Cache File Formats:
        descriptions.csv:
            - movieId (int): Movie identifier
            - description (str): Movie plot/description
            
        movie_scores.csv:
            - movieId (int): Movie identifier
            - imdbId (str): IMDb identifier
            - imdb_rating (float): Rating on 0-10 scale
            - raw_popularity (int): Raw IMDb vote count
            - normalized_popularity (int): Normalized score 0-100

    Notes:
        - Progress message shown every 10 seconds during processing
        - Data retrieval priority:
            1. Use cached data if available
            2. Attempt IMDb API fetch for missing data
            3. Fall back to LLM for missing descriptions
        - Cache file handling:
            * Missing descriptions stored as empty strings
            * Missing numerical values stored as None
            * Files created if they don't exist
        - Text cleaning:
            * Descriptions have newlines removed
            * Leading/trailing whitespace stripped
            * Single-line format enforced

    Example:
        >>> combined_df = pd.DataFrame({
        ...     'movieId': [1, 2],
        ...     'title': ['The Matrix', 'Inception'],
        ...     'imdbId': ['0133093', '1375666']
        ... })
        >>> model_name = "gpt-4o-mini"
        >>> chat_format = "{prompt}"
        >>> base_path = os.path.join('Datasets', 'Movie_Lens_Datasets', 'ml-latest-small')
        >>> descriptions_path = os.path.join(base_path, 'descriptions.csv')
        >>> scores_path = os.path.join(base_path, 'movie_scores.csv')
        >>> url = "https://api.openai.com/v1/chat/completions"
        >>> headers = {
        ...     "Content-Type": "application/json",
        ...     "Authorization": "Bearer YOUR_API_KEY"
        ... }
        >>> descriptions = retrieve_all_descriptions(
        ...     combined_df, model_name, chat_format, descriptions_path,
        ...     scores_path, url, headers
        ... )
    """
    # Load movie scores
    movie_scores_df = load_movie_scores(scores_path, combined_dataframe['movieId'].max())

    # Load cached descriptions
    cached_descriptions = load_cached_descriptions(descriptions_path, combined_dataframe['movieId'].max())
    
    # Track progress message timing
    last_print_time = time.time()

    # Process each movie
    for index, row in combined_dataframe.iterrows():
        movie_id = row['movieId']
        
        # Skip if before start_movie_id
        if movie_id < start_movie_id:
            continue

        current_time = time.time()
        if current_time - last_print_time >= 10:
            print("Retrieving movie data...")
            last_print_time = current_time

        # Check if we need to fetch new data
        cached_description = cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'].iloc[0]
        current_imdb_id = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'].iloc[0]
        current_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0]
        current_raw_popularity = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'].iloc[0]
        current_normalized_popularity = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'].iloc[0]

        # Check what data needs to be updated
        needs_description = cached_description == ""
        needs_imdb_id = current_imdb_id is None
        needs_rating = current_rating is None
        needs_popularity = current_raw_popularity is None or current_normalized_popularity is None

        # Get movie details if any data is missing
        if needs_description or needs_imdb_id or needs_rating or needs_popularity:
            movie_title = row['title']
            imdb_id = row['imdbId']
            
            movie, rating, popularity = get_movie_with_retries(
                imdb_id, movie_title, model_name, chat_format, url, headers, 
                max_retries, delay_between_attempts
            )
            
            if movie:
                # Update description if needed
                if needs_description:
                    if 'plot' in movie:
                        description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
                    else:
                        description = generate_description_with_few_shot(
                            movie_title, model_name, chat_format, url, headers
                        )
                    description = description.replace('\n', ' ').replace('\r', ' ').strip()
                    cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

                # Update IMDb ID if needed
                if needs_imdb_id:
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'] = imdb_id

                # Update rating if needed
                if needs_rating and rating is not None:
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'] = rating

                # Update popularity if needed
                if needs_popularity and popularity is not None:
                    normalized_popularity = normalize_popularity_score(popularity)
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'] = popularity
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'] = normalized_popularity
            
            elif needs_description:
                description = generate_description_with_few_shot(
                    movie_title, model_name, chat_format, url, headers
                )
                description = description.replace('\n', ' ').replace('\r', ' ').strip()
                cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

        # Save periodically
        if (index + 1) % 100 == 0:
            save_cached_descriptions(cached_descriptions, descriptions_path)
            save_movie_scores(movie_scores_df, scores_path)

    # Final save
    save_cached_descriptions(cached_descriptions, descriptions_path)
    save_movie_scores(movie_scores_df, scores_path)

    return cached_descriptions

def generate_all_user_preferences(
    ratings_dataframe: pd.DataFrame,
    combined_dataframe: pd.DataFrame,
    movie_scores_df: pd.DataFrame,
    model_name: str,
    chat_format: str,
    preferences_path: str,
    url: str,
    headers: Dict[str, str],
    start_user_id: int = 1
) -> None:
    """
    Generates and updates user preferences based on rating history for all users in the dataset.

    This function processes each user's movie ratings to generate:
    1.  A text description of their movie preferences using an LLM.
    2.  A preferred movie release date range based on their rated movies.
    3.  Preferences for highly-rated and popular movies based on IMDb scores.

    The function implements caching to avoid regenerating existing preferences and
    saves progress periodically to prevent data loss.

    Parameters:
        ratings_dataframe (pd.DataFrame):
            DataFrame containing user movie ratings with columns:
            -   userId (int): User identifier
            -   movieId (int): Movie identifier
            -   rating (float): User's rating (0.5-5.0 scale)
            -   timestamp (int): Rating timestamp

        combined_dataframe (pd.DataFrame):
            DataFrame containing movie metadata with columns:
            -   movieId (int): Movie identifier
            -   title (str): Movie title with year
            -   description (str): Movie plot description
            -   imdbId (str): IMDb identifier

        movie_scores_df (pd.DataFrame):
            DataFrame containing IMDb metadata with columns:
            -   movieId (int): Movie identifier
            -   imdb_rating (float): IMDb rating (0-10 scale)
            -   normalized_popularity (int): Normalized popularity score (0-100)

        model_name (str):
            Name of the language model to use for generating preference descriptions

        chat_format (str):
            Template string for formatting prompts to the language model

        preferences_path (str):
            Full file path to save the preferences CSV file.  Example: `os.path.join(base_path, 'preferences.csv')`

        url (str):
            API endpoint URL for the language model service

        headers (Dict[str, str]):
            HTTP headers for API requests (e.g., authorization tokens)

        start_user_id (int, optional):
            User ID from which to start processing. Defaults to 1.

    Returns:
        None
            Saves preferences to CSV file at preferences_path

    Data Processing:
        For each user:
        1.  Checks if preferences are already cached.
        2.  If not cached:
            a.  Gets user's top 10 rated movies.
            b.  Uses LLM to generate preference description.
            c.  Calculates preferred date range:
                -   Extracts years from rated movie titles.
                -   Rounds down to the first year of the decade for the minimum year.
                -   Rounds up to the first year of the next decade for the maximum year,
                    or the current year, whichever is smaller.
            d.  Sets rating preference based on average IMDb rating:
                -   prioritize_ratings = True if avg IMDb rating >= 7.0
            e.  Sets popularity preference based on average popularity:
                -   prioritize_popular = True if avg popularity >= 80
        3.  Saves to cache every 100 users processed.

    Cache File Format:
        CSV with columns:
        -   userId (int): User identifier
        -   preferences (str): LLM-generated preference description
        -   date_range (tuple): (start_year, end_year) for preferred movies
        -   prioritize_ratings (bool): Whether user prefers highly-rated movies
        -   prioritize_popular (bool): Whether user prefers popular movies

    Notes:
        -   Skips processing for users with complete cached preferences.
        -   Handles missing IMDb ratings by excluding those movies from averages.
        -   Rounds down to the first year of the decade for the minimum year.
        -   Rounds up to the first year of the next decade for the maximum year,
            or the current year, whichever is smaller.
        -   Saves progress every 100 users and after completion.
        -   Prints calculated date ranges for debugging purposes.

    Example:
        ```python
        >>> ratings_df = pd.DataFrame({
        ...     'userId': [1, 1, 2],
        ...     'movieId': [1, 2, 1],
        ...     'rating': [5.0, 4.0, 3.0]
        ... })
        >>> generate_all_user_preferences(
        ...     ratings_df, combined_df, scores_df,
        ...     "gpt-4o-mini", "{prompt}",
        ...     os.path.join(base_path, "preferences.csv"), "https://api.example.com",
        ...     {"Authorization": "Bearer xyz"}
        ... )
        ```
    """

    # Load existing user preferences from the CSV file, ensuring all user IDs are present
    preferences_df = load_user_preferences(preferences_path, ratings_dataframe['userId'].max())

    # Iterate over each user ID starting from the specified start_user_id
    for user_id in range(start_user_id, ratings_dataframe['userId'].max() + 1):

        # Check if preferences for this user are already cached
        user_preferences_cached = preferences_df.loc[preferences_df['userId'] == user_id, 'preferences'].iloc[0]
        date_range_cached = preferences_df.loc[preferences_df['userId'] == user_id, 'date_range'].iloc[0]
        prioritize_ratings_cached = preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_ratings'].iloc[0]
        prioritize_popular_cached = preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_popular'].iloc[0]

        if user_preferences_cached and date_range_cached and prioritize_ratings_cached is not None and prioritize_popular_cached is not None:
            continue

        # Get all ratings for the current user
        user_ratings = ratings_dataframe[ratings_dataframe['userId'] == user_id]
        
        # Select the top 10 rated movies for the user
        top_rated_movies = user_ratings.nlargest(10, 'rating')
        
        # Prepare a list of tuples containing movie information for the top-rated movies
        rated_movies = [
            (row['movieId'], combined_dataframe.loc[combined_dataframe['movieId'] == row['movieId'], 'title'].values[0], combined_dataframe.loc[combined_dataframe['movieId'] == row['movieId'], 'description'].values[0], row['rating'])
            for index, row in top_rated_movies.iterrows()
        ]

        # Generate user preferences based on the top-rated movies with descriptions if not already cached
        if not user_preferences_cached:
            user_preferences = generate_preferences_from_rated_movies(rated_movies, model_name, chat_format, url, headers)
            preferences_df.loc[preferences_df['userId'] == user_id, 'preferences'] = user_preferences

        # Calculate the date range if not already cached
        if not date_range_cached:
            # Extract years from movie titles
            years = [extract_year_from_title(title) for _, title, _, _ in rated_movies]
            years = [year for year in years if year is not None]

            if years:
                min_year = min(years)
                max_year = max(years)

                # Round down to the first year of the decade for the min year
                min_year = (min_year // 10) * 10

                # Round up to the first year of the next decade for the max year
                current_year = datetime.datetime.now().year
                max_year = ((max_year // 10) + 1) * 10
                max_year = min(max_year, current_year)  # Ensure max_year does not exceed the current year

                date_range = (min_year, max_year)
                print(f"User {user_id}: Calculated date range: {date_range}")  # Debugging line

                idx = preferences_df.loc[preferences_df['userId'] == user_id].index
                if len(idx) == 1:
                    preferences_df.at[idx[0], 'date_range'] = date_range

        # Calculate prioritize_ratings if it's still set to None
        if prioritize_ratings_cached is None:
            # Get IMDb ratings for their top rated movies
            ratings_data = []
            for row in top_rated_movies.itertuples():
                movie_rating = movie_scores_df.loc[movie_scores_df['movieId'] == row.movieId, 'imdb_rating'].iloc[0]
                if movie_rating is not None and movie_rating > 0:  # Only include valid ratings
                    ratings_data.append(movie_rating)
            
            if ratings_data:  # If we found any valid ratings
                # Calculate average IMDb rating
                avg_imdb_rating = sum(ratings_data) / len(ratings_data)
                # Set ratings preference based on average
                preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_ratings'] = (avg_imdb_rating >= 7.0)
            else:
                # Default to False if no valid ratings found
                preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_ratings'] = False

        # Calculate prioritize_popular if it's still set to None
        if prioritize_popular_cached is None:
            # Get popularity scores for their top rated movies
            popularity_data = []
            for row in top_rated_movies.itertuples():
                movie_popularity = movie_scores_df.loc[movie_scores_df['movieId'] == row.movieId, 'normalized_popularity'].iloc[0]
                if movie_popularity is not None:  # Only include valid popularity scores
                    popularity_data.append(movie_popularity)
            
            if popularity_data:  # If we found any valid popularity scores
                # Calculate average popularity
                avg_popularity = sum(popularity_data) / len(popularity_data)
                # Set popularity preference based on average
                preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_popular'] = (avg_popularity >= 80)
            else:
                # Default to False if no valid popularity scores found
                preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_popular'] = False

        # Save the preferences to the CSV file every 100 users
        if user_id % 100 == 0:
            save_user_preferences(preferences_df, preferences_path)

    # Always save the preferences at the end
    save_user_preferences(preferences_df, preferences_path)

def extract_year_from_title(title: str) -> Optional[int]:
    r"""
    Extracts the release year from a movie title.

    This function searches for a four-digit year enclosed in parentheses within a movie title.
    It uses a regular expression pattern r'\((\d{4})\)' which matches:
    - An opening parenthesis \(
    - Exactly four digits \d{4}
    - A closing parenthesis \)

    Parameters:
        title (str):
            The movie title to process. Expected format includes the release year 
            in parentheses, e.g., "The Matrix (1999)" or "Inception (2010)".
            The year must be exactly 4 digits.

    Returns:
        Optional[int]:
            - The release year as an integer if a valid year is found in parentheses
            - None if no year is found or if the title parameter is malformed

    Examples:
        >>> extract_year_from_title("The Matrix (1999)")
        1999
        >>> extract_year_from_title("Star Wars: Episode IV - A New Hope (1977)")
        1977
        >>> extract_year_from_title("Movie Without Year")
        None
        >>> extract_year_from_title("Invalid Year (19)")
        None
    """
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return None

def main():
    """
    Main entry point for the LLM Recommendation System application.

    This function orchestrates the entire recommendation system workflow, including:
    1. Command-line argument parsing for different operating modes
    2. LLM model configuration and API setup
    3. Dataset loading and preprocessing
    4. User interaction for adding new users or calculating metrics
    5. Recommendation generation and evaluation

    Command Line Arguments:
        --mode: Operating mode (choices: development, production, generate-data)
            - development: Uses local Kobold API with phi models
            - production: Uses OpenAI API
            - generate-data: Batch generates descriptions and preferences
        --start-movie-id: Starting movie ID for data generation mode
        --start-user-id: Starting user ID for data generation mode

    Operating Modes:
        1. Data Generation Mode (--mode generate-data):
            - Retrieves movie descriptions from IMDb or generates via LLM
            - Generates user preferences based on rating history
            - Updates cache files for descriptions and preferences

        2. Interactive Mode (development/production):
            a) Adding New Users:
                - Collects movie ratings from users
                - Supports fuzzy or manual movie search
                - Generates or collects user preferences
                - Provides recommendations using both SVD and LLM-enhanced methods
                - Calculates hit rates for new users

            b) Metric Calculation (when adding 0 users):
                - Evaluates multiple algorithms (KNN, SVD, SVD++)
                - Calculates hit rates and cumulative hit rates
                - Compares base and LLM-enhanced versions
                - Displays results in a formatted matrix

    Dataset Options:
        1. MovieLens Small:
            - Smaller dataset for quick testing
            - Faster processing time
            - Suitable for development

        2. MovieLens 32M:
            - Larger dataset for production use
            - More accurate recommendations
            - Longer processing time

    Cache Files:
        - descriptions.csv: Movie plot descriptions
        - preferences.csv: User preferences and settings
        - movie_scores.csv: IMDb ratings and popularity scores

    API Configuration:
        Production Mode:
            - Uses OpenAI API
            - Requires OPENAI_API_KEY environment variable
            - Supports gpt-4o and gpt-4o-mini models

        Development Mode:
            - Uses local Kobold API
            - Supports Phi-3 and Phi-4 models
            - No API key required

    Example Usage:
        # Development mode with default settings
        python script.py

        # Production mode with OpenAI
        python script.py --mode production

        # Generate data starting from specific IDs
        python script.py --mode generate-data --start-movie-id 100 --start-user-id 50

    Notes:
        - Supports both collaborative filtering and content-based recommendations
        - Uses LLMs for enhancing recommendation quality
        - Implements caching to improve performance
        - Provides detailed metrics for system evaluation
    """

    parser = argparse.ArgumentParser(description='LLM Recommendation System')
    parser.add_argument('--mode', type=str, default='development', choices=['development', 'production', 'generate-data'],
                        help='Running mode: development, production, or generate-data')
    parser.add_argument('--start-movie-id', type=int, default=1, help='Starting movie ID for data generation')
    parser.add_argument('--start-user-id', type=int, default=1, help='Starting user ID for data generation')
    args = parser.parse_args()

    if args.mode == 'production':

        # Use OpenAPI 
        api_url = "https://api.openai.com/v1/chat/completions"
        api_key = os.getenv("OPENAI_API_KEY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        openai_models = {
            "gpt-4o": "{prompt}",
            "gpt-4o-mini": "{prompt}"
        }

        # Select OpenAI model here
        model_name = list(openai_models.keys())[1]
        print(model_name)

        # Get the model's corresponding expected chat format
        chat_format = openai_models[model_name]
    else: 
        # Use Kobold's local OpenAI compatible API 
        api_url = "http://localhost:5001/v1/chat/completions"
        headers = { "Content-Type": "application/json" }
        kobold_models = {
            "Phi-3-mini-4k-instruct-q4.gguf": "<|user|>\n{prompt} <|end|>\n<|assistant|>",
            "phi-4-Q6_K": "<|im_start|>user<|im_sep|>{prompt}<|im_end|> <|im_start|>assistant<|im_sep|>"
        }

        # Get the desired locally hosted model
        model_name = list(kobold_models.keys())[1] 

        # Get the corresponding chat format
        chat_format = kobold_models[model_name]

    # Explain the differences between the datasets
    print("Please choose a dataset to use for the recommendation system:")
    print("1. MovieLens Small: A smaller dataset with fewer movies and ratings, suitable for quick testing and development.")
    print("2. MovieLens 32M: A larger dataset with more movies and ratings, providing more accurate recommendations but requiring more processing time.")

    # Ask the user to choose a dataset
    dataset_choice = input("Enter 1 for MovieLens Small or 2 for MovieLens 32M: ").strip()

    # Set the file paths based on the user's choice
    if dataset_choice == "1": # User selected the small dataset
        base_path = os.path.join('Datasets', 'Movie_Lens_Datasets', 'ml-latest-small')

    elif dataset_choice == "2": # User selected the normal size dataset
        base_path = os.path.join('Datasets', 'Movie_Lens_Datasets', 'ml-32m')
    else: 
        print("Invalid choice. Defaulting to MovieLens Small.")
        base_path = os.path.join('Datasets', 'Movie_Lens_Datasets', 'ml-latest-small')

    # Contains userId, movieId, rating, timestamp
    ratings_path = os.path.join(base_path, 'ratings.csv')

    # Contains movieId, title, genres
    movies_path = os.path.join(base_path, 'movies.csv')
    
    # Contains movieId, imdbId, tmdbId. Essentially this serves as a mapping from MovieLens's movieID to Internet Movie Database's and The Movie Database's movie ids.
    links_path = os.path.join(base_path, 'links.csv')
   
    # Contains userId, movieId, tag, timestamp. Allows us to see what keywords and phrases users associated with different movies. 
    # This can allow us to better understand the content of movies when analyzing user preferences.
    tags_path = os.path.join(base_path, 'tags.csv')

    # Contains movieId, description. This allows for quicker description retrieval and will be filled out as we retrieve descriptions from IMDb or generate them.
    descriptions_path = os.path.join(base_path, 'descriptions.csv')

    # Define the path to the preferences CSV file
    preferences_path = os.path.join(base_path, 'preferences.csv')

    # Define the path to the movie scores CSV file
    scores_path = os.path.join(base_path, 'movie_scores.csv')

    # Load the ratings dataset into a pandas dataframe
    ratings_dataframe = pd.read_csv(ratings_path)

    # Load the movies data from the CSV file
    movies_dataframe = pd.read_csv(movies_path)

    # Load the links dataset into a pandas dataframe
    links_dataframe = pd.read_csv(links_path)
    # print(links_dataframe.head())

    # Merge the dataframes on 'movieId'
    combined_dataframe = pd.merge(movies_dataframe, links_dataframe, on='movieId')
    # print(combined_dataframe.head())

    # Load existing user preferences once
    preferences_df = load_user_preferences(preferences_path, ratings_dataframe['userId'].max())

    # Load or initialize movie scores
    movie_scores_df = load_movie_scores(scores_path, movies_dataframe['movieId'].max())
    
    # Create id to title and title to id mappings
    id_to_title, title_to_id = create_movie_mappings(movies_dataframe)

    # Create IMDB id to MovieLens ID mapping and vice versa
    movielens_to_imdb, imdb_to_movielens = create_id_mappings(links_dataframe)

    # Check for generate-data mode
    if args.mode == 'generate-data':
        # Retrieve all descriptions and get the updated DataFrame
        cached_descriptions = retrieve_all_descriptions(combined_dataframe, model_name, chat_format, descriptions_path, api_url, headers, args.start_movie_id)
        
        # Merge descriptions into combined_dataframe
        combined_dataframe = combined_dataframe.merge(cached_descriptions, on='movieId', how='left')
        
        # Update the call to generate_all_user_preferences to use the updated combined_dataframe
        generate_all_user_preferences(ratings_dataframe, combined_dataframe, movie_scores_df, model_name, chat_format, preferences_path, api_url, headers, movie_scores_df, args.start_user_id)
        return

    # Ask how many users to add
    while True:
        try:
            num_users = int(input("How many users would you like to add? (Enter 0 to calculate metrics on dataset): "))
            if num_users >= 0:
                break
            else:
                print("Please enter a number greater than or equal to 1.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    if num_users > 0:
        new_user_ids = []
        all_new_ratings = []

        for user_num in range(1, num_users + 1):
            print(f"\nCollecting data for User {user_num}:")

            # Assign a new user ID
            new_user_id = ratings_dataframe['userId'].max() + 1
            new_user_ids.append(new_user_id)

            # Ask how many movies the user wants to rate
            while True:
                try:
                    n = int(input("How many movies would you like to rate? (Minimum 3): "))
                    if n >= 3:
                        break
                    else:
                        print("Please enter a number greater than or equal to 3.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

            # Step 1: Ask user for 5 movies they love
            new_ratings = []

            # Ask the user if they want to use fuzzy matching or manual search
            search_method = input("Do you want to use fuzzy matching or manually search for each movie? (fuzzy/manual): ").strip().lower()

            print(f"\nPlease enter {n} movies you love and rate them so we can learn about what you like:")

            while len(new_ratings) < n:
                movie_title = input(f"Movie {len(new_ratings) + 1} title: ")
                             
                # Get the IMDbId, director, cover URL, and IMDb title based on the selected search method
                if search_method == 'fuzzy':
                    imdb_id, director, cover_url, imdb_title, imdb_rating, popularity = get_imdb_id_by_title(movie_title, model_name, chat_format, api_url, headers, fetch_full_details=True)
                else:
                    imdb_id, director, cover_url, imdb_title, imdb_rating, popularity = get_imdb_id_by_title(movie_title, model_name, chat_format, api_url, headers, manual_selection=True, fetch_full_details=True)

                if imdb_id:

                    # Convert the found IMDbId to an integer
                    imdb_id = int(imdb_id)

                    # Convert IMDbId to MovieLens movieId
                    movielens_id = imdb_to_movielens.get(imdb_id, None)
                    if movielens_id:
                        # Map MovieLens MovieId back to title
                        confirmed_title = id_to_title.get(movielens_id, None)
                        if confirmed_title:

                            # Display all movie information
                            print(f"IMDb Title: {imdb_title}")
                            print(f"MovieLens Title: {confirmed_title}")
                            print(f"Director: {director}")
                            print(f"IMDb Rating: {imdb_rating:.1f}/10")
                
                            # Calculate normalized popularity
                            normalized_popularity = normalize_popularity_score(popularity)

                            print(f"Raw Popularity Rank: {popularity}")  # Lower number = more popular
                            print(f"Normalized Popularity Score: {normalized_popularity}/100")  # Higher number = more popular
                            print(f"Full Size Cover URL: {cover_url if cover_url else 'No cover image available.'}")
                    
                            # Confirm with user    
                            confirmation = input(f"Is this the movie you want to rate? (yes/no): ").strip().lower()
                            if confirmation == "yes":
                                while True:
                                    try:
                                        rating_input = input(f"Rating for '{movie_title}' (0.5 stars - 5.0 stars): ")
                                        rating = float(rating_input)
                                        if 0.5 <= rating <= 5.0:
                                            break
                                        else:
                                            print("Please enter a rating between 0.5 and 5.0.")
                                    except ValueError:
                                        print("Invalid input. Please enter a numeric value for the rating.")
                                # Get the current timestamp
                                current_timestamp = int(time.time())

                                # Create a new rating entry
                                new_rating = {
                                    'userId': new_user_id,
                                    'movieId': movielens_id,
                                    'rating': rating,
                                    'timestamp': current_timestamp,
                                    'imdb_rating': imdb_rating,
                                    'popularity': popularity

                                }
                                new_ratings.append(new_rating)
                                print(f"Added rating for '{confirmed_title}'.")
                            else:
                                print("Please try another movie.")
                        else:
                            print(f"Could not confirm the title for MovieLens MovieId: {movielens_id}. Please try another movie.")
                    else:
                        print(f"Movie '{movie_title}' with IMDb ID '{imdb_id}' does not exist in the MovieLens dataset of the current size. Please try another movie.")
                else:
                    print(f"Could not find IMDb ID for movie '{movie_title}'. Please try another movie.")

            # Extend the list of all new ratings
            all_new_ratings.extend(new_ratings)

            # Create a dataframe for the new ratings using the list of dictionaries
            new_ratings_df = pd.DataFrame(new_ratings)

            # Concatenate the new ratings DataFrame with the existing ratings DataFrame
            ratings_dataframe = pd.concat([ratings_dataframe, new_ratings_df], ignore_index=True)

            # After collecting movie ratings, ask the user if they want to describe their preferences.
            describe_preferences = input("\nWould you like to describe your movie preferences? (yes/no): ").strip().lower()
        
            date_range = None

            if describe_preferences == "no":
                
                # Calculate average IMDb rating and popularity 
                avg_imdb_rating = sum([float(rating['imdb_rating']) for rating in new_ratings]) / len(new_ratings)
                avg_popularity = sum([float(rating['popularity']) for rating in new_ratings]) / len(new_ratings)

                # Automatically set preferences if average IMDb rating and popularity are high
                prioritize_ratings = avg_imdb_rating >= 7
                prioritize_popular = avg_popularity >= 80

                # Prepare data for fetching descriptions
                rated_movies = [(rating['movieId'], id_to_title[rating['movieId']], rating['rating']) for rating in new_ratings]

                # Fetch movie descriptions for all newly rated movies
                movie_descriptions = get_movie_descriptions(rated_movies, combined_dataframe, model_name, chat_format, descriptions_path, scores_path, api_url, headers, max_retries=5, delay_between_attempts=1)

                # Prepare data for preference generation
                rated_movies_with_descriptions = [
                    (movie_id, title, movie_descriptions[movie_id], rating) for movie_id, title, rating in rated_movies
                ]

                # Generate preferences using the rated movies
                user_preferences = generate_preferences_from_rated_movies(rated_movies_with_descriptions, model_name, chat_format, api_url, headers)
            else:
                # Get user input for preferences
                user_preferences = input("Please describe what kind of movie experiences you are looking for (1 or 2 sentences):\n")

                # Ask about preferences for release date, ratings, and popularity
                # Date range preference
                specify_date_range = input("Would you like to specify a preferred release date range for recommended movies? (yes/no): ").strip().lower()
                if specify_date_range == "yes":
                    start_year = int(input("Enter the earliest release date year preferred: "))
                    end_year = int(input("Enter the latest release date year preferred: "))
                    date_range = (start_year, end_year)

                # Rating preference
                prioritize_ratings = input("Would you like to prioritize movies with high IMDb ratings? (yes/no): ").strip().lower() == 'yes'
                
                # Popularity preference
                prioritize_popular = input("Would you like to prioritize popular movies? (yes/no): ").strip().lower() == 'yes'

            # Automatically determine the date range if not provided
            if date_range is None:
                years = [extract_year_from_title(id_to_title[rating['movieId']]) for rating in new_ratings]
                years = [year for year in years if year is not None]
                if years:
                    min_year = min(years)
                    max_year = max(years)

                    # Round down to the first year of the decade for the min year
                    min_year = (min_year // 10) * 10

                    # Round up to the first year of the next decade for the max year
                    current_year = datetime.datetime.now().year
                    max_year = ((max_year // 10) + 1) * 10
                    max_year = min(max_year, current_year)  # Ensure max_year does not exceed the current year

                    date_range = (min_year, max_year)

            # Output the generated or user-provided preferences
            print("\nUser Preferences:")
            print(user_preferences)
            print(f"Preferred Release Date Range: {date_range[0]} - {date_range[1]}")
            print(f"Prioritize High IMDb Ratings: {'Yes' if prioritize_ratings else 'No'}")
            print(f"Prioritize Popular Movies: {'Yes' if prioritize_popular else 'No'}")

            # Create a temporary dataframe for the new preference
            new_preference = pd.DataFrame([{
                'userId': new_user_id, 
                'preferences': user_preferences, 
                'date_range': date_range,
                'prioritize_ratings': prioritize_ratings,
                'prioritize_popular': prioritize_popular
            }])

            # Concat temp dataframe to the full preferences dataframe
            preferences_df = pd.concat([preferences_df, new_preference], ignore_index=True)

            # Save the updated preferences DataFrame
            save_user_preferences(preferences_df, preferences_path)

        # Define a Reader with the appropriate rating scale
        reader = Reader(rating_scale=(0.5, 5.0))

        # The dataframe must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings in this order. 
        # The reader object is also required with the rating_scale parameter specified.
        data = Dataset.load_from_df(ratings_dataframe[['userId', 'movieId', 'rating']], reader)

        # Create an SVD algorithm instance
        algo = SVD()

        # Prepare to calculate cumulative hit rate for the new users using SVD
        print("\nCalculating cumulative hit rate...")

        # Number of favorite movies to use for similarity score calculation
        num_favorites = 3

        cumulative_hit_rate_svd_10, cumulative_hit_rate_llm_10 = calculate_cumulative_hit_rate(
            algo, ratings_dataframe, id_to_title, combined_dataframe, model_name, chat_format,
            descriptions_path, api_url, headers, preferences_path, scores_path, n=10, threshold=4.0, user_ids=new_user_ids, use_llm=True, 
            num_favorites=num_favorites, search_count=100
        )
        
        '''
        cumulative_hit_rate_svd_100, cumulative_hit_rate_llm_100 = calculate_cumulative_hit_rate(
            algo, ratings_dataframe, id_to_title, combined_dataframe, model_name, chat_format,
            descriptions_path, api_url, headers, preferences_path, scores_path, n=100, threshold=4.0, user_ids=new_user_ids, use_llm=True, search_count=1000
        )
        '''
        
        print(f"\nCalculating Hit Rate for new users (SVD, threshold=4.0, n=10): {cumulative_hit_rate_svd_10:.2f}")
        print(f"Cumulative Hit Rate for new users (LLM-enhanced, threshold=4.0, n=10): {cumulative_hit_rate_llm_10:.2f}")
    
        '''
        print(f"\nCalculating Hit Rate for new users (SVD, threshold=4.0, n=100): {cumulative_hit_rate_svd_100:.2f}")
        print(f"Cumulative Hit Rate for new users (LLM-enhanced, threshold=4.0, n=100): {cumulative_hit_rate_llm_100:.2f}")
        '''
        
        # Now, generate recommendations for each user using both methods
        all_movie_ids = movies_dataframe['movieId'].unique()

        # Build the full trainset
        full_trainset = data.build_full_trainset()

        # Train SVD on the full trainset
        algo.fit(full_trainset)

        for user_id in new_user_ids:
            print(f"\nRecommendations for User {user_id}:")

            # Set number of recommendations to generate with SVD, we will pick the top ten to compare with LLM
            n_times_10 = 100

            if n_times_10 >= 10:
                n = int(n_times_10 / 10)
            else:
                n = n_times_10

            # Get top n * 10 with traditional algorithm
            top_n_for_user_extended = get_top_n_recommendations(algo, user_id, all_movie_ids, ratings_dataframe, id_to_title, n_times_10)

            # Print the top N recommendations for the user with estimated ratings
            print(f"\nTop {n} recommendations according to SVD:")
            for movie_id, movie_title, est_rating in top_n_for_user_extended[:n]:
                print(f"Movie Title: {movie_title}, Estimated Rating: {est_rating:.2f}")

            # Get this user's preferences
            user_preferences = preferences_df.loc[preferences_df['userId'] == user_id, 'preferences'].iloc[0]

            # Get the user's preferred movie release date range
            user_date_range = preferences_df.loc[preferences_df['userId'] == user_id, 'date_range'].iloc[0]

            # Get movie descriptions for the top N * 10 movies
            movie_descriptions = get_movie_descriptions(top_n_for_user_extended, combined_dataframe, model_name, chat_format,
                                                        descriptions_path, scores_path, api_url, headers, max_retries=5, delay_between_attempts=1)

            # Get the user's favorite movies
            favorite_movie_titles = get_user_favorite_movies(user_id, ratings_dataframe, id_to_title, num_favorites=num_favorites)

            # When calling find_top_n_similar_movies, get the user's preferences from preferences_df
            prioritize_ratings = preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_ratings'].iloc[0]
            prioritize_popular = preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_popular'].iloc[0]

            # Get recommendations using LLM-enhanced method
            top_n_similar_movies = find_top_n_similar_movies(
                user_preferences, 
                movie_descriptions, 
                id_to_title, 
                model_name, 
                chat_format, 
                n, 
                api_url, 
                headers, 
                movie_scores_df=movie_scores_df,
                prioritize_ratings=prioritize_ratings,
                prioritize_popular=prioritize_popular,  
                favorite_movies=favorite_movie_titles,
                num_favorites=num_favorites,
                date_range=user_date_range
            )

            # Print the best movie recommendations according to the traditional algorithm enhanced by the LLM
            print(f"\nTop {n} recommendations according to LLM-enhanced method:")
            for movie_id, score in top_n_similar_movies:
                movie_title = id_to_title[movie_id]
                print(f"Movie Title: {movie_title}, Similarity Score: {score}")
    else:

        print("\nCalculating metrics for all algorithms...")

        # Initialize algorithms
        algorithms = {
            'KNN': KNNBasic(),
            'SVD': SVD(),
            'SVD++': SVDpp()
        }

        # Define metrics configuration
        n_values = [1, 5, 10]
        num_favorites = 3
        threshold = 4.0

        # Initialize results matrix
        results = {
            'Metric': [
                f'Hit Rate N@{n}' for n in n_values
            ] + [
                f'Cumulative Hit (>= {threshold}) N@{n}' for n in n_values
            ]
        }

        # Calculate metrics for each algorithm
        for algo_name, algo in algorithms.items():
            print(f"\nProcessing {algo_name}...")
            
            # Initialize lists to store results for this algorithm
            base_hit_rates = []
            llm_hit_rates = []
            base_cum_hit_rates = []
            llm_cum_hit_rates = []

            # Calculate metrics for each N value
            for n in n_values:
                # Calculate base and LLM-enhanced metrics (no threshold)
                hit_rate, llm_hit_rate = calculate_cumulative_hit_rate(
                    algo, ratings_dataframe, id_to_title, combined_dataframe,
                    model_name, chat_format, descriptions_path, api_url, headers,
                    preferences_path, scores_path, n=n, threshold=0.0,  # No threshold
                    user_ids=None, use_llm=True, num_favorites=num_favorites, search_count=100
                )
                base_hit_rates.append(f"{hit_rate:.4f}")
                llm_hit_rates.append(f"{llm_hit_rate:.4f}")

                # Calculate base and LLM-enhanced metrics (with threshold)
                cum_hit_rate, llm_cum_hit_rate = calculate_cumulative_hit_rate(
                    algo, ratings_dataframe, id_to_title, combined_dataframe,
                    model_name, chat_format, descriptions_path, api_url, headers,
                    preferences_path, scores_path, n=n, threshold=threshold,
                    user_ids=None, use_llm=True, num_favorites=num_favorites, search_count=100
                )
                base_cum_hit_rates.append(f"{cum_hit_rate:.4f}")
                llm_cum_hit_rates.append(f"{llm_cum_hit_rate:.4f}")

            # Add results to matrix
            results[algo_name] = base_hit_rates + base_cum_hit_rates  # Regular hit rates + cumulative hit rates
            results[f'{algo_name} LLM'] = llm_hit_rates + llm_cum_hit_rates  # LLM-enhanced versions

        # Create and display the matrix using tabulate
        headers = ['Metric', 'KNN', 'SVD', 'SVD++', 'KNN LLM', 'SVD LLM', 'SVD++ LLM']
        table_data = [[metric] + [results[algo].get(i, '') for algo in headers[1:]] 
                    for i, metric in enumerate(results['Metric'])]
        
        print("\nResults Matrix:")
        print(tabulate(table_data, headers=headers, tablefmt='pipe', stralign='center'))

if __name__ == "__main__":
    main()