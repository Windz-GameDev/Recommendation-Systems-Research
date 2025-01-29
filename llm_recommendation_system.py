from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd

# Used for extracting similarity scores from LLM string responses
import re

# Used for handling API requests to LLM
import requests

# This library proves a simple and  complete API for acccessing IMDB data
from imdb import Cinemagoer, IMDbError 

import math

import logging

# Set the logging level for the IMDbPY library
logging.getLogger('imdbpy').setLevel(logging.ERROR)
logging.getLogger('imdbpy').disabled = True

import time
import math

# Used to select a random movie to drop for use as the test set when calculating hit rate
import random
import argparse
import os

# Used for generating accurate movie release date preferences
import datetime

def normalize_popularity_score(votes, max_rank=1000000):
    """
    Normalize IMDb vote count to a 0-100 scale.
    
    Parameters:
    - votes (int/float): Number of votes from IMDb
    - max_rank (int): Maximum expected number of votes (default: 1,000,000)
    
    Returns:
    - int: Normalized popularity score from 0-100, where:
        - 100 = extremely popular (votes >= max_rank)
        - 0 = not popular (no votes)
        - None if votes is None
    """
    if votes is None or votes <= 0:
        return None
        
    normalized = max(0, min(100, round(100 * (votes / max_rank))))
    return normalized

def calculate_cumulative_hit_rate(algo, ratings_dataframe, id_to_title, combined_dataframe, model_name, chat_format, 
                                  descriptions_path, api_url, headers, preferences_path, scores_path, n=10, threshold=4.0, 
                                  user_ids=None, use_llm=False, max_retries=5, delay_between_attempts=1, num_favorites=3):
    '''
    Calculate the cumulative hit rate of the recommendation algorithm using leave-one-out cross-validation.

    If user_ids is provided, the hit rate is calculated only for those users. If user_ids is None, the hit rate 
    is calculated for all users.

    Parameters:
    - algo: The trained recommendation algorithm.
    - ratings_dataframe: The dataframe containing all ratings.
    - id_to_title: Dictionary mapping movieId to title.
    - combined_dataframe: The dataframe containing movie information, including IMDb IDs.
    - model_name: The name of the language model to use for generating descriptions.
    - chat_format: The format string to structure the few-shot examples and movie title.
    - descriptions_path: The path to the descriptions CSV file.
    - api_url: The API endpoint URL to send the request to.
    - headers: The headers to include in the API request.
    - preferences_path: The path to the preferences CSV file.
    - n: The number of top recommendations to consider for each user.
    - threshold: The rating threshold to consider an item as relevant.
    - user_ids: List of user IDs to calculate hit rate for, or None to calculate for all users.
    - use_llm: Boolean flag to determine whether to use LLM-enhanced recommendations.
    - max_retries: Maximum number of retries for fetching movie data.
    - delay_between_attempts: Delay between retry attempts in seconds.
    - num_favorites: The number of favorite movies to use for similarity score calculation.

    Returns:
    - hit_rate: The cumulative hit rate.
    '''
    # Create a copy of the ratings DataFrame which will have the test set ratings removed
    ratings_dataframe_testset_removed = ratings_dataframe.copy()

    # Initialize the leave-one-out test set
    loo_testset = []

    # Get unique user IDs from the ratings dataset
    if user_ids is None:
        unique_user_ids = ratings_dataframe_testset_removed['userId'].unique()
    else:
        unique_user_ids = user_ids

    # Create the leave-one-out testset by removing one random rating for each user
    for user_id in unique_user_ids:
        user_ratings = ratings_dataframe_testset_removed[ratings_dataframe_testset_removed['userId'] == user_id]
        if len(user_ratings) > 1:
            test_rating_index = random.randint(0, len(user_ratings) - 1)
            test_rating = user_ratings.iloc[test_rating_index]
            loo_testset.append((user_id, test_rating['movieId'], test_rating['rating']))
            ratings_dataframe_testset_removed = ratings_dataframe_testset_removed.drop(user_ratings.index[test_rating_index])
    
    # Define a Reader with the appropriate rating scale
    reader = Reader(rating_scale=(0.5, 5.0))

    # Create the train set from the remaining ratings
    data = Dataset.load_from_df(ratings_dataframe_testset_removed[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Train the algorithm on the trainset
    algo.fit(trainset)

    # Get a list of all movie IDs from the ratings dataset
    all_movie_ids = ratings_dataframe['movieId'].unique()

    # Initialize hit count
    base_hit_count = 0
    llm_hit_count = 0 
    total_count = 0

    # Load user preferences once for all users
    max_user_id = ratings_dataframe['userId'].max()
    preferences_df = load_user_preferences(preferences_path, max_user_id)

    # Load movie scores for ratings/popularity
    movie_scores_df = load_movie_scores(scores_path, combined_dataframe['movieId'].max())

    # Iterate over each user in the leave-one-out test set
    for user_id, movie_id, rating in loo_testset:

        # Check if the left-out rating is about the threshold
        if rating >= threshold:
            total_count += 1

            # Get top N recommendations for the user using the base algorithm
            top_n_recommendations = get_top_n_recommendations(algo, user_id, all_movie_ids, ratings_dataframe_testset_removed, id_to_title, n)
            recommended_movie_ids = [rec_movie_id for rec_movie_id, _, _ in top_n_recommendations]

            # Check if the left-out movie is in the top N recommendations for the base algorithm
            if movie_id in recommended_movie_ids:
                base_hit_count += 1

            ## LLM-enhanced recommendations
            if use_llm:

                # Get initial recommendations using SVD
                top_n_times_x_for_user = get_top_n_recommendations(algo, user_id, all_movie_ids, ratings_dataframe_testset_removed, id_to_title, n*10)

                # Get user's rating/popularity preferences
                prioritize_ratings = preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_ratings'].iloc[0]
                prioritize_popular = preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_popular'].iloc[0]

                # Get user preferences
                user_preferences = preferences_df.loc[preferences_df['userId'] == user_id, 'preferences'].iloc[0]

                # Get the user's date range
                user_date_range = preferences_df.loc[preferences_df['userId'] == user_id, 'date_range'].iloc[0]
                
                # Get movie descriptions
                movie_descriptions = get_movie_descriptions(top_n_times_x_for_user, combined_dataframe, model_name, chat_format,
                                                            descriptions_path, scores_path, api_url, headers, max_retries, delay_between_attempts)

                # Get the user's favorite movies
                favorite_movie_titles = get_user_favorite_movies(user_id, ratings_dataframe_testset_removed, id_to_title, num_favorites=num_favorites)

                # Find top N similar movies using LLM with rating/popularity preferences
                top_n_similiar_movies = find_top_n_similar_movies(user_preferences, movie_descriptions,
                                                               id_to_title, model_name, chat_format, n,
                                                               api_url, headers, movie_scores_df,
                                                               prioritize_ratings, prioritize_popular,
                                                               favorite_movie_titles, num_favorites,
                                                               date_range=user_date_range)

                llm_recommended_movie_ids = [rec_movie_id for rec_movie_id, _ in top_n_similiar_movies]

                # Check if the left-out movie is in the top N recommendations for the LLM-enhanced algorithm
                if movie_id in llm_recommended_movie_ids:
                    llm_hit_count += 1

    # Calculate hit rate
    base_hit_rate = base_hit_count / total_count if total_count > 0 else 0
    llm_hit_rate = llm_hit_count / total_count if total_count > 0 else 0

    return base_hit_rate, llm_hit_rate if use_llm else None

def get_movie_scores(movie_id, imdb_id, movie_title):
    """
    Get IMDb rating and popularity score for a movie/TV show.
    """
    ia = Cinemagoer()
    
    try:
        movie = ia.get_movie(imdb_id)
        if movie:
            rating = movie.get('rating', None)
            votes = movie.get('votes', None)
            normalized_popularity = normalize_popularity_score(votes) if votes else None
            
            return rating, normalized_popularity, votes
            
    except IMDbError as e:
        print(f"Failed to get scores for {movie_title}: {e}")
        
    return None, None, None

def load_movie_scores(scores_path, max_movie_id):
    """
    Load movie scores from CSV, initializing new rows with None if needed.
    
    Parameters:
    - scores_path (str): Path to the scores CSV file
    - max_movie_id (int): Maximum movie ID to accommodate
    
    Returns:
    - pd.DataFrame: DataFrame containing:
        - movieId: MovieLens ID of the movie 
        - imdbId: IMDb ID (None if unknown)
        - imdb_rating: IMDb rating 0-10 (None if unavailable)
        - normalized_popularity: Popularity score 0-100 (None if unavailable)
        - raw_popularity: Raw IMDb popularity rank (None if unavailable)
    """
    if os.path.exists(scores_path):
        # Read the CSV file
        movie_scores = pd.read_csv(scores_path)
        
        # Replace NaN, empty strings, and 0 values with None
        movie_scores = movie_scores.replace({
            '': None,
            0: None,
            'nan': None,
            float('nan'): None
        })
        
        # Ensure all movie IDs from 1 to max_movie_id are present
        all_movie_ids = pd.DataFrame({'movieId': range(1, max_movie_id + 1)})
        complete_scores = pd.merge(all_movie_ids, movie_scores, on='movieId', how='left')
        
        # Replace any remaining NaN values with None
        complete_scores = complete_scores.where(pd.notnull(complete_scores), None)
        
    else:
        # Create new DataFrame with all movie IDs and None values
        complete_scores = pd.DataFrame({
            'movieId': range(1, max_movie_id + 1),
            'imdbId': [None] * max_movie_id,
            'imdb_rating': [None] * max_movie_id,
            'normalized_popularity': [None] * max_movie_id,
            'raw_popularity': [None] * max_movie_id
        })
    
    return complete_scores

def save_movie_scores(scores_df, scores_path):
    """
    Save movie scores to CSV file.
    
    Parameters:
    - scores_df (pd.DataFrame): DataFrame containing movie scores
    - scores_path (str): Path to save the CSV file
    """
    try:
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        scores_df.to_csv(scores_path, index=False)
    except (IOError, OSError) as e:
        print(f"Error saving movie scores: {e}")


# Function to create mapping dictionaries between MovieLens Movie IDs and Movie titles
def create_movie_mappings(movies_df):
    """
        Create mapping dictionaries between MovieLens Movie IDs and Movie titles.

        Parameters:
        - movies_df: DataFrame containing movie information with 'movieId' and 'title' columns.

        Returns:
        - id_to_title: Dictionary mapping movieId to title.
        - title_to_id: Dictionary mapping title to movieId.
    """
    # Create a dictionary mapping movieId to title
    id_to_title = pd.Series(movies_df.title.values, index=movies_df.movieId).to_dict()

    # Create a dictionary mapping title to movieId
    title_to_id = pd.Series(movies_df.movieId.values, index=movies_df.title).to_dict()

    return id_to_title, title_to_id

def create_id_mappings(links_df):
    """
    Create mapping dictionaries between MovieLens Movie IDs and IMDb IDs.

    Parameters: 
    - links_df: DataFrame containing movie information with 'movieId' and 'imdbId' columns.

    Returns:
    - movielens_to_imdb: Dictionary mapping MovieLens movieId to IMDbId.
    - imdb_to_movielens: Dictionary mapping IMDbId to MovieLens movieId.
    """

    # Create a dictionary mapping MovieLens movieId to IMDbId
    movielens_to_imdb = pd.Series(links_df.imdbId.values, index=links_df.movieId).to_dict()

    # Create a dictionray mapping IMDbId to MovieLens movieId
    imdb_to_movielens = pd.Series(links_df.movieId.values, index=links_df.imdbId).to_dict()
    
    return movielens_to_imdb, imdb_to_movielens


def evaluate_model(algo, testset, data):
    """
        Evaluate the performance of a recommendation algorithm.

        Parameters:
        - algo: The trained recommendation algorithm.
        - testset: The test set for evaluating the algorithm.
        - data: The full dataset for cross-validation.

        Returns:
        - None
    """
    print(f"Evaluating {algo} algorithm on the testset")
    predictions = algo.test(testset)

    # Compute and print RMSE
    rmse = accuracy.rmse(predictions)
    print("RMSE is a measure of the differences between predicted and actual ratings. "
        "It gives higher weight to larger errors, making it sensitive to outliers.")

    # Compute and print MAE
    mae = accuracy.mae(predictions)
    print("MAE is the average of the absolute differences between predicted and actual ratings. "
        "It provides a straightforward measure of prediction accuracy.")

    # Perform cross-validation
    print("Performing cross-validation...")
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

def get_top_n_recommendations(algo, user_id, all_movie_ids, ratings_dataframe, id_to_title, n=10):
    """
    Get top N movie recommendations for a user.

    Parameters:
    - algo: The trained recommendation algorithm.
    - user_id: The ID of the user for whom to generate recommendations.
    - all_movie_ids: List of all movie IDs in the dataset.
    - ratings_dataframe: The dataframe used to generate predictions. 
    - id_to_title: Dictionary mapping movieId to title.
    - n: Number of top recommendations to return.

    Returns:
    - top_n: List of tuples containing movie IDs and their estimated ratings.
    """

    # Get the set of movies the user has already rated (excluding the test set if called from the calculate hit rate function)
    rated_movies = set(ratings_dataframe[ratings_dataframe['userId'] == user_id]['movieId'])

    # Predict ratings for all movies the user hasn't rated yet
    predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movies]

    # Sort the predictions by the estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top N movie IDs and their estimated ratings
    top_n = [(pred.iid, id_to_title[pred.iid], pred.est) for pred in predictions[:n]]

    return top_n

def retrieve_all_descriptions(combined_dataframe, model_name, chat_format, descriptions_path, scores_path, url, headers, start_movie_id=1, max_retries=5, delay_between_attempts=1):
    """
    Retrieves and caches movie descriptions and IMDb scores (ratings and popularity) for all movies in the dataset.
    
    This function processes each movie in the combined_dataframe to:
    1. Check if description/scores exist in cache 
    2. Fetch missing data from IMDb if needed
    3. Generate descriptions via LLM if IMDb fetch fails
    4. Normalize popularity scores to 0-100 scale
    5. Save data to cache files periodically

    Parameters:
    - combined_dataframe (pd.DataFrame): DataFrame containing movie metadata including IMDb IDs
    - model_name (str): Name of LLM model for generating descriptions
    - chat_format (str): Format string for LLM prompts 
    - descriptions_path (str): Path to descriptions cache CSV file
    - scores_path (str): Path to IMDb scores cache CSV file
    - url (str): LLM API endpoint URL  
    - headers (dict): HTTP headers for API requests
    - start_movie_id (int): Movie ID to start processing from (default: 1)
    - max_retries (int): Maximum retries for IMDb API calls (default: 5)
    - delay_between_attempts (int): Delay in seconds between retries (default: 1)

    Returns:
    - pd.DataFrame: DataFrame containing all movie descriptions

    Notes:
    - Saves descriptions and scores every 100 movies processed 
    - Displays progress message every 10 seconds
    - Uses IMDb API for scores and descriptions with fallback to LLM generation
    - For ratings/popularity, uses None for missing values instead of 0
    - Caches all data to avoid repeated API calls
    - Handles both missing descriptions and scores independently
    """
    
    # Load data
    movie_scores_df = load_movie_scores(scores_path, combined_dataframe['movieId'].max())
    cached_descriptions = load_cached_descriptions(descriptions_path, combined_dataframe['movieId'].max())
    
    # Track progress
    last_print_time = time.time()

    # Process each movie
    for index, row in combined_dataframe.iterrows():
        movie_id = row['movieId']
        if movie_id < start_movie_id:
            continue

        # Check for missing data using explicit None check
        cached_description = cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'].iloc[0]
        current_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0]
        needs_scores = current_rating is None

        # Get movie details if needed
        if cached_description == "" or needs_scores:
            movie_title = row['title']
            imdb_id = row['imdbId']
            
            movie, rating, popularity = get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, url, headers, max_retries, delay_between_attempts)
            
            if movie:
                # Update description if needed 
                if cached_description == "":
                    if 'plot' in movie:
                        description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
                    else:
                        description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                    description = description.replace('\n', ' ').replace('\r', ' ').strip()
                    cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

                # Update scores if needed
                if needs_scores:
                    normalized_popularity = normalize_popularity_score(popularity) if popularity else None

                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'] = imdb_id
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'] = rating if rating is not None else None
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'] = normalized_popularity
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'] = popularity if popularity is not None else None
            
            elif cached_description == "":
                description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
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

def generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers):
    """
    Generate a movie description using few-shot prompting with a language model.
    
    Parameters:
    - movie_title (str): The title of the movie.
    - model_name (str): The name of the model to use for generating the description.
    - chat_format (str): The format string to structure the few-shot examples and movie title.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.

    Returns: 
    = str: The generated movie description.
    """

    role_instruction = (
        "You are a helpful assistant that generates movie descriptions. "
        "Do not use newlines in the description."
        "The instructions and examples exist to help you understand the context, do not include them in your response. "
        "End the response immediately after you finish your description and include nothing else."
    )

    # Few-shot examples
    few_shot_examples = (
        "#### START EXAMPLES ####\n"
        "Example 1:\n"
        "Movie title: Inception\n"
        "Description: A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the project and his team to disaster.\n"
        "Example 2:\n"
        "Movie title: The Matrix\n"
        "Description: A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.\n"
        "#### END EXAMPLES ####\n"
    )

    # Format the prompt using the provided chat format 
    prompt = chat_format.format(prompt=few_shot_examples + "Now, generate a description for the following movie without using newlines:\n" + f"Movie title: {movie_title}\nDescription:")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": role_instruction},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 120,
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
    else:
        return ""
    
def get_imdb_id_by_title(title, model_name, chat_format, url, headers, manual_selection=False, results_limit=20, page_limit=5, fetch_full_details=False):
    """
    Retrieve the IMDb ID for a movie given its title.

    If manual_selection is True, the user will be asked to pick from the top few search results 
    instead of letting the system do fuzzy matching automatically.

    Parameters:
    - title (str): The title of the movie.
    - model_name (str): The name of the model to use for generating the similarity score (currently used in existing fuzzy matching logic).
    - chat_format (str): A format string to structure the user input and movie description.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.
    - manual_selection (bool): Whether to prompt the user manually for a selection among the search results.
    - results_limit (int): Maximum number of search results to display when manual_selection is True.
    - page_limit (int): Maximum number of search results to display per page when manual_selection is True.

    Returns:
    - tuple: A tuple containing the IMDb ID of the movie (str), the director (str), the cover URL (str), and the IMDb title (str), or (None, None, None, None) if not found.
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

            # Check if the result is a movie
            # if movie.get('kind') == 'movie':
            
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

        few_shot_examples = (
            "#### START EXAMPLES ####\n"
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
            "#### END EXAMPLES ####\n"
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
                    {"role": "system", "content": "You are a helpful assistant that determines if a movie title matches the user's input. You always only respond with numbers between 0 and 1."},
                    {"role": "user", "content": full_prompt_formatted}
                ],
                "max_tokens": 5,
                "temperature": 0
            }

            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                content = response.json().get("choices", [])[0].get("message", {}).get("content", "0")
                try:
                    similarity_score = float(content)
                    # print(f"Similarity score for '{movie_title}': {similarity_score}")
                except ValueError: 
                    print(f"Could not convert similarity score to float: {content}")
                    similarity_score = -1 # Assign a default low similarity score

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
                popularity = full_movie.get('popularity', 0.0)                
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

def get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, url, headers, max_retries=10, delay=1):
    """
    Attempts to retrieve movie/TV show data from IMDb with appropriate popularity metrics.
    """
    ia = Cinemagoer()
    attempt = 0
    movie = None
    rating = None 
    popularity = None

    while attempt < max_retries:
        try:
            # Get the movie/show with all info
            movie = ia.get_movie(imdb_id, info=['main', 'plot'])
            
            if movie:
                rating = movie.get('rating', None)
                popularity = movie.get('votes', None)
                    
                if 'plot' in movie:
                    return movie, rating, popularity
                    
        except IMDbError as e:
            if 'HTTPError 404' in str(e):
                print(f"HTTP 404 error encountered for {movie_title}'s IMDb ID {imdb_id}. Attempting to find IMDb ID by title.")
                imdb_id, _, _, _, _, _ = get_imdb_id_by_title(movie_title, model_name, chat_format, url, headers)
                if not imdb_id:
                    print(f"Could not find the IMDb ID with the title '{movie_title}'.")
                    return None, None, None
            else:
                print(f"Attempt {attempt + 1} failed: {e}")

        attempt += 1
        if attempt < max_retries:
            time.sleep(delay)
        
    return None, None, None

def load_cached_descriptions(descriptions_path, max_movie_id):
    """
    Loads movie descriptions from a CSV file and ensures all movie IDs from 1 to max_movie_id are present.

    This function attempts to load movie descriptions from a specified CSV file into a pandas DataFrame.
    If the file exists, it reads the descriptions into the DataFrame. If the file does not exist, it
    initializes a DataFrame with all movie IDs from 1 to max_movie_id, with empty strings as placeholders
    for descriptions. This ensures that every movie ID has a corresponding entry, even if the description
    is initially missing.

    Parameters:
    - descriptions_path (str): The path to the descriptions CSV file.
    - max_movie_id (int): The maximum movie ID to ensure all IDs from 1 to this number have entries.

    Returns:
    pandas.DataFrame: A DataFrame containing movie descriptions, with all movie IDs from 1 to max_movie_id
    included. Descriptions are filled with empty strings where data is missing.
    """
    if os.path.exists(descriptions_path):
        cached_descriptions = pd.read_csv(descriptions_path)
        # Ensure all movie IDs from 1 to max_movie_id are present
        all_movie_ids = pd.DataFrame({'movieId': range(1, max_movie_id + 1)})
        complete_cached_descriptions = pd.merge(all_movie_ids, cached_descriptions, on='movieId', how='left')
        complete_cached_descriptions['description'] = complete_cached_descriptions['description'].fillna("")
    else:
        # Create a DataFrame with all movie IDs from 1 to max_movie_id
        complete_cached_descriptions = pd.DataFrame({'movieId': range(1, max_movie_id + 1), 'description': [""] * max_movie_id})

    return complete_cached_descriptions
    
def save_cached_descriptions(cached_descriptions, descriptions_path):
    """
    Saves the cached movie descriptions to a CSV file.

    Parameters:
    - cached_descriptions (pandas.DataFrame): A DataFrame containing movie descriptions with
      columns 'movieId' and 'description'. Each row represents a movie and its corresponding
      description.
    - descriptions_path (str): The path to the descriptions CSV file.
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

def get_movie_descriptions(top_n_movies, combined_dataframe, model_name, chat_format, descriptions_path, scores_path, url, headers, max_retries, delay_between_attempts):
    """
    Retrieves movie descriptions and scores for a list of top N movies using their IMDb IDs.

    This function attempts to retrieve both descriptions and scores (rating, popularity) for a given list of top N movies.
    It first checks if data is available in the caches. If not, it tries to fetch the data from IMDb.
    If the description is unavailable, it generates one using few-shot prompting with a language model.
    All new data is cached for future use.

    Parameters:
    - top_n_movies (list): A list of tuples, each containing a movie ID, movie title, and estimated rating.
    - combined_dataframe (pandas.DataFrame): A DataFrame containing movie information, including IMDb IDs.
    - model_name (str): The name of the language model to use for generating descriptions.
    - chat_format (str): The format string to structure the few-shot examples and movie title.
    - descriptions_path (str): The path to the descriptions CSV file.
    - scores_path (str): The path to the scores CSV file.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.
    - max_retries (int): The maximum number of retry attempts for fetching movie data.
    - delay_between_attempts (int): The delay in seconds between retry attempts.
    
    Returns:
    - descriptions (dict): A dictionary mapping movie IDs to their descriptions.

    Notes:
    - The function uses a caching mechanism to store and retrieve data, reducing the need
      for repeated API calls or description generation.
    - If a description is not found in the cache, the function attempts to retrieve it from IMDb.
    - If the IMDb retrieval fails, a description is generated using the language model.
    - New data is appended to the respective cache files for future use.
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
        
        # Get the current rating value
        current_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0]

        # Get the IMDb ID for the current movie
        imdb_id = combined_dataframe.loc[combined_dataframe['movieId'] == movie_id, 'imdbId'].values[0]
        

        # If we need to fetch either description or scores, make the IMDb API call
        if cached_description == "" or current_rating is None:
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
                        
                # Update scores if rating is None
                if current_rating is None:
                    normalized_popularity = normalize_popularity_score(votes) if votes else None
                    
                    # Only update ratings and scores if they're valid
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'] = imdb_id
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'] = rating 
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'] = normalized_popularity
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'] = votes
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

def find_top_n_similar_movies(user_input, movie_descriptions, id_to_title, model_name, chat_format, n, url, headers, 
                            movie_scores_df=None, prioritize_ratings=False, prioritize_popular=False,
                            favorite_movies=None, num_favorites=3, max_retries=3, date_range=None):
    """
    Finds the top N most similar movies to the user's input from a dictionary mapping of movie ids to descriptions using a Large Language Model (LLM).

    It constructs a prompt for the LLM using few-shot examples to guide the model in generating a similarity score for each movie. 
    The function attempts to extract a similarity score from the LLM's response and convert it to a float. 
    If that fails, the function uses a regular expression to extract a number from the LLM's response.
    If there is no number in the response, it retries by generating another response. 
    If no valid number is found after the specified number of max_retries, a default similarity score of 0.0 is used.
    Once a similarity score has been calculated for each movie id in movie_descriptions dictionary, the function returns the top N movies with the highest similarity scores.

    Parameters:
    - user_input (str): A string describing the user's movie preferences.
    - movie_descriptions (dict): A dictionary mapping movie IDs to their descriptions. Each description provides a brief overview of the movie's content.
    - id_to_title (dict): A dictionary mapping movie IDs to their titles. This is used to include the movie title in the prompt for the LLM.
    - model_name (str): The name of the LLM to use for generating similarity scores. This should correspond to a model available at the specified API endpoint.
    - chat_format (str): A format string used to structure the prompt for the LLM. It should include a placeholder for the prompt content.
    - n (int): The number of top similar movies to return.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.
    - favorite_movies (list of str, optional): A list of the user's favorite movies to include in the prompt. Defaults to None.
    - num_favorites (int, optional): The number of favorite movies to include in the prompt. Defaults to 3.
    - max_retries (int, optional): The maximum number of retries to attempt if the LLM response cannot be converted to a float. Defaults to 3.
    - date_range (tuple, optional): A tuple containing the start and end year of the user's preferred movie date range. Defaults to None.

    Returns:
    - top_n_movies (list of tuples): A list of tuples containing the movie ID and similarity score of the top N most similar movies.
    """

    similarity_scores = []

    # Define the role and instructions with rating scale explanation
    role_instruction = (
        "You are a movie recommendation assistant. "
        "Your task is to evaluate how well a movie description aligns with a user's stated preferences and their favorite movies."
        "Always respond with a number between -1.0 and 1.0, where:\n"
        "-1.0 means the movie goes completely against their preferences,\n"
        "0 means neutral or there isn't enough information,\n"
        "1.0 is a perfect match."
    )

    # Add example input and output to guide the LLM
    few_shot_examples = (
        "#### START EXAMPLES ####\n"
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
        "#### END EXAMPLES ####\n"
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
            "#### USER INPUT ####\n"
            f"User input: {user_input} "
            f"{'I prefer movies with high IMDb ratings. ' if prioritize_ratings else ''}"
            f"{'I prefer popular/trending movies.' if prioritize_popular else ''}\n"
            f"{favorite_movies_prompt}"
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
                {"role": "system", "content": role_instruction},
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

def get_user_favorite_movies(user_id, ratings_dataframe, id_to_title, num_favorites=3):
    """
    Retrieves a user's favorite movies based on their ratings in the `ratings_dataframe` by selecting the top `num_favorites` rated movies using the `nlargest` method.


    Parameters:
    - user_id (int): The ID of the user.
    - ratings_dataframe (pandas.DataFrame): The DataFrame containing user ratings.
    - id_to_title (dict): A dictionary mapping movie IDs to their titles.
    - num_favorites (int): The number of favorite movies to retrieve (default is 3).

    Returns:
    - list: A list of the user's favorite movie titles.
    """
    user_ratings = ratings_dataframe[ratings_dataframe['userId'] == user_id]
    favorite_movies = user_ratings.nlargest(num_favorites, 'rating')['movieId'].tolist()
    favorite_movie_titles = [id_to_title[movie_id] for movie_id in favorite_movies]
    return favorite_movie_titles
    
def generate_preferences_from_rated_movies(rated_movies, model_name, chat_format, url, headers):
    
    """
    Generate a user preference summary using few-shot prompting with a language model.

    Parameters:
    - rated_movies (list of tuples): Each tuple contains (movie_id, movie_title, description, rating).
    - model_name (str): The name of the model to use for generating the description.
    - chat_format (str): The format string to structure the few-shot examples and movie title.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.

    Returns:
    - str: The generated user preference summary.
    """

    max_tokens = 60

    role_instruction = (
        "You are a helpful assistant that generates comprehensive user preference summaries based on multiple movies that a user has rated. "
        "Ensure the preferences are written in the first person without newlines. "
        "The instructions and examples exist to help you understand the context and how to format your response, do not include them in your response. "
        "You should match the format of the user preferences responses in the examples exactly with nothing else in your response."
    )

    # Few-shot examples with real movies and aggregated preferences
    few_shot_examples = (
        "#### START EXAMPLES ####\n"
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
        "#### END EXAMPLES ####\n"
    )

    # Combine rated movie descriptions with titles and ratings
    combined_descriptions = "\n".join(
        f"Movie title: {title}\nDescription: {description}\nUser rating: {rating}"
        for _, title, description, rating in rated_movies
    )

    # Format the prompt using the provided chat format
    prompt = chat_format.format(prompt=few_shot_examples + f"Now, based on the following movie titles, descriptions, and ratings, using {max_tokens} tokens or less, generate a comprehensive user preference summary in the first person, without newlines:\n" + combined_descriptions)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": role_instruction},
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

        # Further post-process to remove any unintended content
        if "#### START EXAMPLES ####" in summary:
            summary = summary.split("#### START EXAMPLES ####")[0].strip()

        return summary
    except requests.exceptions.RequestException as e:
        print("An error occurred while making the request:", e)
        return

def load_user_preferences(preferences_path, max_user_id):
    """
    Loads user preferences from a CSV file and ensures all user IDs from 1 to max_user_id are present.

    Parameters:
    - preferences_path (str): Path to the CSV file storing user preferences
    - max_user_id (int): Maximum user ID to ensure all IDs from 1 to this number have entries

    Returns:
    - pandas.DataFrame: DataFrame containing:
        - userId: User ID
        - preferences: Text description of user's movie preferences 
        - date_range: Tuple of (min_year, max_year) for preferred movie release dates
        - prioritize_ratings: Boolean indicating if user prefers high-rated movies
        - prioritize_popular: Boolean indicating if user prefers popular movies

    Notes:
    - Creates empty entries for missing user IDs with blank preferences and None values
    - Handles both existing and new preference files
    - Ensures backwards compatibility if popularity/rating columns don't exist
    """
    if os.path.exists(preferences_path):
        preferences_df = pd.read_csv(preferences_path)
        # Ensure all user IDs from 1 to max_user_id are present
        all_user_ids = pd.DataFrame({'userId': range(1, max_user_id + 1)})
        complete_preferences_df = pd.merge(all_user_ids, preferences_df, on='userId', how='left')
        complete_preferences_df['preferences'] = complete_preferences_df['preferences'].fillna("")
        
        # Initialize columns if they don't exist
        if 'date_range' in complete_preferences_df.columns:
            complete_preferences_df['date_range'] = complete_preferences_df['date_range'].fillna(value="")
            complete_preferences_df['date_range'] = complete_preferences_df['date_range'].apply(lambda x: None if x == "" else x)
        else:
            complete_preferences_df['date_range'] = None
            
        if 'prioritize_ratings' in complete_preferences_df.columns:
            complete_preferences_df['prioritize_ratings'] = complete_preferences_df['prioritize_ratings'].apply(lambda x: None if pd.isna(x) else x)
        else:
            complete_preferences_df['prioritize_ratings'] = None
            
        if 'prioritize_popular' in complete_preferences_df.columns:
            complete_preferences_df['prioritize_popular'] = complete_preferences_df['prioritize_popular'].apply(lambda x: None if pd.isna(x) else x)
        else:
            complete_preferences_df['prioritize_popular'] = None
    else:
        # Create a DataFrame with all user IDs from 1 to max_user_id
        complete_preferences_df = pd.DataFrame({
            'userId': range(1, max_user_id + 1),
            'preferences': [""] * max_user_id,
            'date_range': [None] * max_user_id,
            'prioritize_ratings': [None] * max_user_id,
            'prioritize_popular': [None] * max_user_id
        })

    return complete_preferences_df

def save_user_preferences(preferences_df, preferences_path):
    """
    Save the entire preferences DataFrame to a CSV file with retry functionality.

    Parameters:
    - preferences_df (pandas.DataFrame): The DataFrame containing all user preferences.
    - preferences_path (str): The path to the CSV file where preferences will be saved.

    Returns:
    - None
    """
    max_retries = 10

    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(preferences_path), exist_ok=True)
            preferences_df.to_csv(preferences_path, index=False, encoding='utf-8')
            print("Preferences saved successfully.")
            break
        except (IOError, OSError) as e:
            print(f"Attempt {attempt + 1}: Error saving preferences: {e}")
            if attempt < max_retries - 1:
                print("Please close any applications using the file and press Enter to retry...")
                input()
            else:
                print("Failed to save the file after multiple attempts. Please ensure the file is not open in another application.")

def retrieve_all_descriptions(combined_dataframe, model_name, chat_format, descriptions_path, scores_path, url, headers, start_movie_id=1, max_retries=5, delay_between_attempts=1):
    """
    Retrieves and caches movie descriptions and IMDb scores (ratings and popularity) for all movies in the dataset.
    
    This function processes each movie in the combined_dataframe to:
    1. Check if description/scores exist in cache
    2. Fetch missing data from IMDb if needed
    3. Generate descriptions via LLM if IMDb fetch fails
    4. Normalize popularity scores to 0-100 scale
    5. Save data to cache files periodically
    
    Parameters:
    - combined_dataframe (pd.DataFrame): DataFrame containing movie metadata including IMDb IDs
    - model_name (str): Name of LLM model for generating descriptions
    - chat_format (str): Format string for LLM prompts
    - descriptions_path (str): Path to descriptions cache CSV file
    - scores_path (str): Path to IMDb scores cache CSV file  
    - url (str): LLM API endpoint URL
    - headers (dict): HTTP headers for API requests
    - start_movie_id (int): Movie ID to start processing from (default: 1)
    - max_retries (int): Maximum retries for IMDb API calls (default: 5)
    - delay_between_attempts (int): Delay in seconds between retries (default: 1)

    Returns:
    - pd.DataFrame: DataFrame containing all movie descriptions, indexed by MovieLens ID

    Notes:
    - Saves descriptions and scores every 100 movies processed
    - Displays progress message every 10 seconds
    - Uses IMDb API for scores and descriptions with fallback to LLM generation
    - Caches all data to avoid repeated API calls
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

        current_rating = movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'].iloc[0]    

        # Get movie details if needed
        if cached_description == "" or current_rating is None:
            movie_title = row['title']
            imdb_id = row['imdbId']
            
            movie, rating, popularity = get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, url, headers, max_retries, delay_between_attempts)
            
            if movie:
                # Update description if needed
                if cached_description == "":
                    if 'plot' in movie:
                        description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
                    else:
                        description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
                    description = description.replace('\n', ' ').replace('\r', ' ').strip()
                    cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

                # Update scores if current rating is None
                if current_rating is None:
                    normalized_popularity = normalize_popularity_score(popularity)

                    # Only update ratings and scores if they're valid
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdbId'] = imdb_id
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'imdb_rating'] = rating 
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'normalized_popularity'] = normalized_popularity
                    movie_scores_df.loc[movie_scores_df['movieId'] == movie_id, 'raw_popularity'] = popularity
            
            elif cached_description == "":
                description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
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

def generate_all_user_preferences(ratings_dataframe, combined_dataframe, movie_scores_df, model_name, chat_format, preferences_path, url, headers, start_user_id=1):
    """
    Generates and updates preferences for all users based on their rating history.

    Parameters:
    - ratings_dataframe (pd.DataFrame): User movie ratings data
    - combined_dataframe (pd.DataFrame): Movie metadata including descriptions
    - movie_scores_df (pd.DataFrame): IMDb ratings and popularity scores for movies  
    - model_name (str): Name of LLM model to use
    - chat_format (str): Prompt template for LLM
    - preferences_path (str): Path to save preferences
    - url (str): LLM API endpoint
    - headers (dict): API request headers
    - start_user_id (int): User ID to start processing from

    The function:
    1. Loads existing preferences
    2. For each user:
        - Gets their top 10 rated movies
        - Generates preference text using LLM if not cached
        - Calculates preferred date range from rated movies
        - Sets rating/popularity preferences based on average scores
        - Saves progress every 100 users

    Notes:
    - Skips users with complete cached preferences
    - Prioritizes high ratings if avg IMDb rating >= 7.0 
    - Prioritizes popularity if avg popularity >= 80
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

        # Calculate prioritize_ratings and prioritize_popular if they are still set to None
        if prioritize_ratings_cached is None or prioritize_popular_cached is None:
            # Get IMDb ratings and popularity scores for their top rated movies
            top_rated_movies_data = []
            for row in top_rated_movies.itertuples():
                movie_rating = movie_scores_df.loc[movie_scores_df['movieId'] == row.movieId, 'imdb_rating'].iloc[0]
                movie_popularity = movie_scores_df.loc[movie_scores_df['movieId'] == row.movieId, 'normalized_popularity'].iloc[0]
                if movie_rating > 0:  # Only include movies that have IMDb ratings
                    top_rated_movies_data.append((movie_rating, movie_popularity))
            
            if top_rated_movies_data:  # If we found any valid ratings
                # Calculate averages
                avg_imdb_rating = sum(rating for rating, _ in top_rated_movies_data) / len(top_rated_movies_data)
                avg_popularity = sum(popularity for _, popularity in top_rated_movies_data) / len(top_rated_movies_data)
                
                # Set preferences based on their favorite movies' scores
                preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_ratings'] = (avg_imdb_rating >= 7.0)
                preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_popular'] = (avg_popularity >= 80)
            else:
                # Default to False if no valid ratings found
                preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_ratings'] = False
                preferences_df.loc[preferences_df['userId'] == user_id, 'prioritize_popular'] = False

        # Save the preferences to the CSV file every 100 users
        if user_id % 100 == 0:
            save_user_preferences(preferences_df, preferences_path)

    # Always save the preferences at the end
    save_user_preferences(preferences_df, preferences_path)

# Function to extract the year from a movie title
def extract_year_from_title(title):
    """
    Extracts the release year from a movie title.

    The function searches for a four-digit year enclosed in parentheses within the movie title.
    If a year is found, it is returned as an integer. If no year is found, the function returns None.

    Parameters:
    - title (str): The movie title, which may include the release year in parentheses.

    Returns:
    - int or None: The release year as an integer if found, otherwise None.
    """
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return None

def main():

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
            "Phi-3-mini-4k-instruct-q4.gguf": "<|user|>\n{prompt} <|end|>\n<|assistant|>"
        }

        # Get the desired locally hosted model
        model_name = list(kobold_models.keys())[0] 

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
            num_users = int(input("How many users would you like to add? (Enter 0 to calculate cumulative hit rate): "))
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
            num_favorites=num_favorites
        )
        
        '''
        cumulative_hit_rate_svd_100, cumulative_hit_rate_llm_100 = calculate_cumulative_hit_rate(
            algo, ratings_dataframe, id_to_title, combined_dataframe, model_name, chat_format,
            descriptions_path, api_url, headers, preferences_path, scores_path, n=100, threshold=4.0, user_ids=new_user_ids, use_llm=True
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

            # Get recommendations using LLM-enhanced method
            top_n_similar_movies = find_top_n_similar_movies(user_preferences, movie_descriptions, id_to_title, model_name, chat_format, n, api_url, headers, 
                                                            movie_scores_df, favorite_movies=favorite_movie_titles, num_favorites=num_favorites, date_range=user_date_range)

            # Print the best movie recommendations according to the traditional algorithm enhanced by the LLM
            print(f"\nTop {n} recommendations according to LLM-enhanced method:")
            for movie_id, score in top_n_similar_movies:
                movie_title = id_to_title[movie_id]
                print(f"Movie Title: {movie_title}, Similarity Score: {score}")
    else:
        # Create an SVD algorithm instance
        algo = SVD()

        # Proceed to calculate cumulative hit rates for all users
        print("\nCalculating cumulative hit rates for all users...")

        # Define the list of n values
        n_values = [10, 15, 20, 100]

        # Number of favorites to use for similarity scoring
        num_favorites = 3

        # Loop over each n value
        for n in n_values:
            cumulative_hit_rate_svd, cumulative_hit_rate_llm = calculate_cumulative_hit_rate(
                algo, ratings_dataframe, id_to_title, combined_dataframe, model_name, chat_format,
                descriptions_path, api_url, headers, preferences_path, scores_path, n=n, threshold=4.0, user_ids=None, use_llm=True,
                num_favorites=num_favorites
            )
            print(f"Cumulative Hit Rate for all users (SVD, threshold=4.0, n={n}): {cumulative_hit_rate_svd:.4f}")
            print(f"Cumulative Hit Rate for all users (LLM-enhanced, threshold=4.0, n={n}): {cumulative_hit_rate_llm:.4f}")

if __name__ == "__main__":
    main()