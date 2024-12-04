from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd

# Used for handling API requests to LLM
import requests

# This library proves a simple and  complete API for acccessing IMDB data
from imdb import Cinemagoer, IMDbError 

import logging

# Set the logging level for the IMDbPY library
logging.getLogger('imdbpy').setLevel(logging.ERROR)
logging.getLogger('imdbpy').disabled = True

import time

# Used to select a random movie to drop for use as the test set when calculating hit rate
import random
import argparse
import os

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

def calculate_cumulative_hit_rate(algo, ratings_dataframe, id_to_title, combined_dataframe, model_name, chat_format, 
                                  descriptions_path, api_url, headers, preferences_path, n=10, threshold=4.0, 
                                  user_ids=None, use_llm=False, max_retries=5, delay_between_attempts=1):
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

                # Get user preferences
                user_preferences = preferences_df.loc[preferences_df['userId'] == user_id, 'preferences'].iloc[0]
                
                # Get movie descriptions
                movie_descriptions = get_movie_descriptions(top_n_times_x_for_user, combined_dataframe, model_name, chat_format,
                                                            descriptions_path, api_url, headers, max_retries, delay_between_attempts)

                # Find top N similiar movies using LLM
                top_n_similiar_movies = find_top_n_similar_movies(user_preferences, movie_descriptions, id_to_title,
                                                                  model_name, chat_format, n, api_url, headers)

                llm_recommended_movie_ids = [rec_movie_id for rec_movie_id, _ in top_n_similiar_movies]

                # Check if the left-out movie is in the top N recommendations for the LLM-enhanced algorithm
                if movie_id in llm_recommended_movie_ids:
                    llm_hit_count += 1

    # Calculate hit rate
    base_hit_rate = base_hit_count / total_count if total_count > 0 else 0
    llm_hit_rate = llm_hit_count / total_count if total_count > 0 else 0

    return base_hit_rate, llm_hit_rate if use_llm else None

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
    
def get_imdb_id_by_title(title, model_name, chat_format, url, headers):
    '''
    Retrieve the IMDb ID for a movie given its title, using LLM for fuzzy matching if necessary.

    Parameters:
    - title (str): The title of the movie.
    - model_name (str): The name of the model to use for generating the similarity score.
    - chat_format (str): A format string to structure the user input and movie description
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.

    Returns:
    - str: The IMDb ID of the movie, or None if not found.
    '''

    ia = Cinemagoer()
    try:
        # Search for the movie by title
        search_results = ia.search_movie(title)
        for movie in search_results:

            #Check if the result is a movie
            if movie.get('kind') == 'movie':
                # Check for an exact title match with case sensitivity
                if movie['title'] == title:
                    imdb_id = movie.movieID
                    # print(f"Found exact match for '{title}': IMDb ID is {imdb_id}")
                    return imdb_id

        # If no exact match is found, use LLM to find the best match
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
            if movie.get('kind') == 'movie':
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
                # print(f"Best match for '{title}' using LLM: {best_match['title']} IMDb ID is {imdb_id}")
                return imdb_id
            else:
                print(f"No match found for title in IMDb movies: {title}")
                return None
    except IMDbError as e:
        print(f"An error occured while searching for movie {title}'s IMDb ID: {e}")
        return None

def get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, url, headers, max_retries=10, delay=1):
    
    """
    Attempts to retrieve a movie object with plot and reviews from IMDb, retrying if necessary.

    Parameters:
    - imdb_id (str): The IMDb ID of the movie to retrieve.
    - max_retries (int): The maximum number of retry attempts. Defaults to 10.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.
    - delay (int): The delay in seconds between retry attempts. Defaults to 1.

    Returns:
    - movie (dict): The movie object containing plot if successful, otherwise None.
    """
    ia = Cinemagoer()
    attempt = 0
    movie = None

    while attempt < max_retries:
        try:
            # Attempt to get the movie with reviews
            movie = ia.get_movie(imdb_id, info=['main','plot'])
            if movie and 'plot' in movie:
                # print(f"The movie data for {movie_title} was retrieved successfully.")
                return movie
        except IMDbError as e:
            # Check if the error is an HTTP 404 error
            if 'HTTPError 404' in str(e):
                print(f"HTTP 404 error encountered for {movie_title}'s IMDd ID {imdb_id}.  Attempting to find IMDb ID by title.")
                imdb_id = get_imdb_id_by_title(movie_title, model_name, chat_format, url, headers)
                if not imdb_id:
                    print(f"Could not find the IMDd ID with the title '{movie_title}'.")
                    return None
            else:
                print(f"Attempt {attempt + 1} failed: {e}")
                print("") # Print nothing right now

        attempt += 1
        if attempt < max_retries:
            # print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            #print(f"Max retries reached. Could not fetch the plot for {movie_title}.")
            return None

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
        complete_cached_descriptions['description'].fillna("", inplace=True)
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

def get_movie_descriptions(top_n_movies, combined_dataframe, model_name, chat_format, descriptions_path, url, headers, max_retries, delay_between_attempts):
    """
    Retrieves movie descriptions for a list of top N movies using their IMDb IDs.

    This function attempts to retrieve movie descriptions for a given list of top N movies.
    It first checks if a description is available in the cache. If not, it tries to fetch the
    description from IMDb. If the description is still unavailable, it generates one using
    few-shot prompting with a language model. All new descriptions are cached for future use.

    Parameters:
    - top_n_movies (list): A list of tuples, each containing a movie ID, movie title, and estimated rating.
    - combined_dataframe (pandas.DataFrame): A DataFrame containing movie information, including IMDb IDs.
    - model_name (str): The name of the language model to use for generating descriptions.
    - chat_format (str): The format string to structure the few-shot examples and movie title.
    - descriptions_path (str): The path to the descriptions CSV file.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.
    - max_retries (int): The maximum number of retry attempts for fetching movie data.
    - delay_between_attempts (int): The delay in seconds between retry attempts.
    
    Returns:
    - descriptions (dict): A dictionary mapping movie IDs to their descriptions.

    Notes:
    - The function uses a caching mechanism to store and retrieve movie descriptions, reducing the need
      for repeated API calls or description generation.
    - If a description is not found in the cache, the function attempts to retrieve it from IMDb using
      the `get_movie_with_retries` function.
    - If the IMDb retrieval fails, a description is generated using the `generate_description_with_few_shot` function.
    - New descriptions are appended to the cache and saved to a CSV file for future use.
    """

    # Initalize an empty dictionary to store movie descriptions for the movies stored top_n_movies
    descriptions = {}

    # Load cached descriptions from a CSV file
    cached_descriptions = load_cached_descriptions(descriptions_path, combined_dataframe['movieId'].max())

    # Track the last time the message was printed
    last_print_time = time.time()

    # Iterate over the list of top N movies
    for movie_id, movie_title, _ in top_n_movies:
        
        current_time = time.time()
        # Check if 10 seconds have passed since the last time the message was printed 
        if current_time - last_print_time >= 10:
            print("Retrieving movie descriptions...")
            last_print_time = current_time

        # Check if we have a cached description for the current movie
        cached_description = cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'].iloc[0]
        if cached_description != "":
            descriptions[movie_id] = cached_description
            continue

        imdb_id = combined_dataframe.loc[combined_dataframe['movieId'] == movie_id, 'imdbId'].values[0]
        movie = get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, url, headers, max_retries, delay_between_attempts)
        if movie and 'plot' in movie:
            description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
        else:
            # Could not find a description, as a last resort, generate a description using few shot prompting
            description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)
            # print(descriptions[movie_id])

        # Ensure the description is a single line
        description = description.replace('\n', ' ').replace('\r', ' ').strip()

        # Store the description in the dictionary
        descriptions[movie_id] = description

        # Replace the empty string descriptions with the new descriptions
        cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

    # Saved the updated cache descriptions back to the CSV file    
    save_cached_descriptions(cached_descriptions, descriptions_path)

    return descriptions

def find_top_n_similar_movies(user_input, movie_descriptions, id_to_title, model_name, chat_format, n, url, headers):
    """
    Finds the top N most similar movies to the user's input from a list of movie descriptions using a Large Language Model (LLM).

    This function leverages an LLM to evaluate the similarity between a user's preferences and a set of movie descriptions. It constructs a prompt for the LLM using few-shot examples to guide the model in generating a similarity score for each movie. The top N movies with the highest similarity scores are returned.

    Parameters:
    - user_input (str): A string describing the user's movie preferences.
    - movie_descriptions (dict): A dictionary mapping movie IDs to their descriptions. Each description provides a brief overview of the movie's content.
    - id_to_title (dict): A dictionary mapping movie IDs to their titles. This is used to include the movie title in the prompt for the LLM.
    - model_name (str): The name of the LLM to use for generating similarity scores. This should correspond to a model available at the specified API endpoint.
    - chat_format (str): A format string used to structure the prompt for the LLM. It should include a placeholder for the prompt content.
    - n (int): The number of top similar movies to return.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.

    Returns:
    - top_n_movies (list of tuples): A list of tuples containing the movie ID and similarity score of the top N most similar movies.
    """

    similarity_scores = []

    # Define the role and instructions with rating scale explanation
    role_instruction = (
        "You are a movie recommendation assistant. "
        "Your task is to evaluate how well a movie description aligns with a user's stated preferences. "
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
        "Movie title: Inception\n"
        "Movie description: A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the project and his team to disaster.\n"
        "Rate how likely you think the movie aligns with the user's interests (respond with a number in range [-1, 1]):\n"
        "0.9\n"
        "Example 2:\n"
        "User input: I enjoy light-hearted comedies with a lot of humor.\n"
        "Movie title: The Dark Knight\n"
        "Movie description: Set within a year after the events of Batman Begins (2005), Batman, Lieutenant James Gordon, and new District Attorney Harvey Dent successfully begin to round up the criminals that plague Gotham City, until a mysterious and sadistic criminal mastermind known only as \"The Joker\" appears in Gotham, creating a new wave of chaos. Batman's struggle against The Joker becomes deeply personal, forcing him to \"confront everything he believes\" and improve his technology to stop him. A love triangle develops between Bruce Wayne, Dent, and Rachel Dawes.\n"
        "Rate how likely you think the movie aligns with the user's interests (respond with a number in range [-1, 1]):\n"
        "-0.7\n"
        "Example 3:\n"
        "User input: I am fascinated by historical documentaries.\n"
        "Movie title: The Lord of the Rings: The Fellowship of the Ring\n"
        "Movie description: A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.\n"
        "Rate how likely you think the movie aligns with the user's interests (respond with a number in range [-1, 1]):\n"
        "-0.5\n"
        "#### END EXAMPLES ####\n"
    )

    for movie_id, description in movie_descriptions.items():

        # Get the title for the prompt
        movie_title = id_to_title[movie_id]


        # Format the prompt using the provided chat format
        prompt_content = (
            "#### USER INPUT ####\n"
            f"User input: {user_input}\n"
            f"Movie title: {movie_title}\n"
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

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            content = response.json().get("choices", [])[0].get("message", {}).get("content", "0")
            # print(content)
            try:
                similarity_score = float(content)
                # Clamp the similarity score between -1 and 1
                if similarity_score < -1.0:
                    similarity_score = -1.0
                elif similarity_score > 1.0:
                    similarity_score = 1.0
            except ValueError:
                print(f"Could not convert similarity score to float for movie '{movie_title}': {content}")
                
                # Assign a similarity score of 0 if conversion fails 
                similarity_score = 0.0

        else: 
            print(f"Request failed for movie '{movie_title}' with status code {response.status_code}")

            # Assign a similarity score of 0 for failed requests
            similarity_score = 0.0

        similarity_scores.append((movie_id, similarity_score))

    # Sort the movies by similarity score in descending order and select the top N
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_n_movies = similarity_scores[:n]
 
    return top_n_movies

def ia_test_function(movie_title, combined_dataframe, model_name, chat_format, url, headers):
    """
    Tests the retrieval of movie information from IMDb for a given movie title.

    Parameters:
    - movie_title (str): The title of the movie to retrieve information for.
    - combined_dataframe (DataFrame): A DataFrame containing movie information, including IMDb IDs.
    - model_name (str): The name of the model to use for generating the similarity score.
    - chat_format (str): A format string to structure the user input.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.

    Returns:
    - movie_description (str): The plot description of the movie, or a default message if not available.
    """
    # Access IMDb ID for a specific movie
    imdb_id = combined_dataframe.loc[combined_dataframe['title'] == movie_title, 'imdbId'].values[0]
    print(f"IMDb ID for '{movie_title}' is {imdb_id}")

    # Get the movie object with title and plot
    movie = get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, url, headers)

    # Access the plot (description)
    movie_description = None
    if movie:
        if isinstance(movie['plot'], list):
            movie_description = movie['plot'][0]  # We are using the first summary for simplicity
            print(f"Movie Description: {movie_description}")
        else:
            movie_description = movie['plot']
    else:
        # Set empty string because movie description could not be found
        movie_description = ""

    # Add the description to the DataFrame
    # combined_dataframe.loc[combined_dataframe['title'] == movie_title, 'description'] = movie_description

    '''

    # This code has been temporarily disabled until review retrieving functionality has been reimplemented

    # Access reviews
    if movie:
        print("First 10 Reviews:")
        for i, review in enumerate(movie['reviews'][:10], start=1):
            print(f"\nReview {i}:")
            print(f"Author: {review['author']}")
            print(f"Date: {review['date']}")
            print(f"Rating: {review.get('rating', 'N/A')}")
            print(f"Content: {review['content']}\n")
    else:
        print("No reviews found for this movie.")
    '''

    return movie_description

def test_api_call(model_name, prompt, chat_format, url, headers):
    """
    Sends a POST request to a specified LLM API endpoint to generate a response based on a user message.

    Parameters:
    - model_name (str): The name of the model to use for generating the response.
    - prompt (str): The message or prompt from the user that needs a response.
    - chat_format (str): A format string to structure the user message. It should contain a placeholder for the user message.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.

    Returns:
    - str: The generated response from the LLM if successful, otherwise None.
    """

    # Format user message using provided chat format
    user_prompt = chat_format.format(prompt=prompt)

    # Define the request payload (data)
    payload = {
        "model": model_name, 
        "messages": [
            {
                "role" : "system",
                "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        "max_tokens": 8,
        "temperature" : 0.7  
    }

    try:
        # Send the POST request with the payload as JSON data
        response = requests.post(url, headers=headers, json=payload)

        # Check if the status was successful
        if response.status_code == 200:
            # Parse the response
            generated_response = response.json().get("choices", [])[0].get("message", {}).get("content", "")
            print("Generated Response:", generated_response)
            return generated_response
        else:
            print("Failed to generate response")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
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

    This function attempts to load user preferences from a specified CSV file into a pandas DataFrame.
    If the file exists, it reads the preferences into the DataFrame. If the file does not exist, it
    initializes a DataFrame with all user IDs from 1 to max_user_id, with empty strings as placeholders
    for preferences. This ensures that every user ID has a corresponding entry, even if the preferences
    are initially missing.

    Parameters:
    - preferences_path (str): The path to the preferences CSV file.
    - max_user_id (int): The maximum user ID to ensure all IDs from 1 to this number have entries.

    Returns:
    pandas.DataFrame: A DataFrame containing user preferences, with all user IDs from 1 to max_user_id
    included. Preferences are filled with empty strings where data is missing.
    """
    if os.path.exists(preferences_path):
        preferences_df = pd.read_csv(preferences_path)
        # Ensure all user IDs from 1 to max_user_id are present
        all_user_ids = pd.DataFrame({'userId': range(1, max_user_id + 1)})
        complete_preferences_df = pd.merge(all_user_ids, preferences_df, on='userId', how='left')
        complete_preferences_df['preferences'].fillna("", inplace=True)
    else:
        # Create a DataFrame with all user IDs from 1 to max_user_id
        complete_preferences_df = pd.DataFrame({'userId': range(1, max_user_id + 1), 'preferences': [""] * max_user_id})

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

def retrieve_all_descriptions(combined_dataframe, model_name, chat_format, descriptions_path, url, headers, start_movie_id=1, max_retries=5, delay_between_attempts=1):
    """
    Retrieve descriptions for all movies starting from a specific movie ID.

    This function iterates over all movies in the combined DataFrame, starting from the specified movie ID.
    It checks if a description is already cached; if not, it attempts to retrieve the description from IMDb.
    If retrieval fails, it generates a description using a language model. The descriptions are saved to a CSV file
    every 100 movies to prevent data loss.

    Parameters:
    - combined_dataframe (pandas.DataFrame): DataFrame containing movie information, including IMDb IDs.
    - model_name (str): The name of the language model to use for generating descriptions.
    - chat_format (str): The format string to structure the few-shot examples and movie title.
    - descriptions_path (str): The path to the descriptions CSV file.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.
    - start_movie_id (int): The movie ID to start processing from. Defaults to 1.
    - max_retries (int): The maximum number of retry attempts for fetching movie data. Defaults to 5.
    - delay_between_attempts (int): The delay in seconds between retry attempts. Defaults to 1.

    Returns:
    - None
    """
    # Load existing descriptions from the CSV file, ensuring all movie IDs are present
    cached_descriptions = load_cached_descriptions(descriptions_path, combined_dataframe['movieId'].max())
    
    # Iterate over each row in the combined DataFrame
    for index, row in combined_dataframe.iterrows():
        
        # Extract the movie ID from the current row
        movie_id = row['movieId']

        # Skip movies with IDs less than the starting movie ID
        if movie_id < start_movie_id:
            continue

        # Check if the description for this movie is already cached
        cached_description = cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'].iloc[0]
        if cached_description != "":
            continue

        # Extract the movie title and IMDb ID from the current row
        movie_title = row['title']
        imdb_id = row['imdbId']

        # Attempt to retrieve the movie data, including the plot, from IMDb
        movie = get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, url, headers, max_retries, delay_between_attempts)
        
        # If the movie data is retrieved and contains a plot, use it as the description
        if movie and 'plot' in movie:
            description = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
        else:
            # If no plot is available, generate a description using a language model
            description = generate_description_with_few_shot(movie_title, model_name, chat_format, url, headers)

        # Update the cached descriptions with the new description for the current movie
        cached_descriptions.loc[cached_descriptions['movieId'] == movie_id, 'description'] = description

        # Save the cached descriptions to the CSV file every 100 movies
        if (index + 1) % 100 == 0:
            save_cached_descriptions(cached_descriptions, descriptions_path)

    # Always save the cached descriptions at the end
    save_cached_descriptions(cached_descriptions, descriptions_path)

    # Return the updated cached descriptions
    return cached_descriptions

def generate_all_user_preferences(ratings_dataframe, combined_dataframe, model_name, chat_format, preferences_path, url, headers, start_user_id=1):
    """
    Generate preferences for all users starting from a specific user ID.

    This function iterates over all users in the ratings DataFrame, starting from the specified user ID.
    It generates user preferences based on the top 10 rated movies for each user. The preferences are saved
    to a CSV file every 100 users to prevent data loss.

    Parameters:
    - ratings_dataframe (pandas.DataFrame): DataFrame containing all movie ratings made by users.
    - combined_dataframe (pandas.DataFrame): DataFrame containing movie information, including IMDb IDs and descriptions.
    - model_name (str): The name of the language model to use for generating preferences.
    - chat_format (str): The format string to structure the few-shot examples and movie title.
    - preferences_path (str): The path to the preferences CSV file.
    - url (str): The API endpoint URL to send the request to.
    - headers (dict): The headers to include in the API request.
    - start_user_id (int): The user ID to start processing from. Defaults to 1.

    Returns:
    - None
    """
    # Load existing user preferences from the CSV file, ensuring all user IDs are present
    preferences_df = load_user_preferences(preferences_path, ratings_dataframe['userId'].max())

    # Iterate over each user ID starting from the specified start_user_id
    for user_id in range(start_user_id, ratings_dataframe['userId'].max() + 1):
        # Check if preferences for this user are already cached
        if preferences_df.loc[preferences_df['userId'] == user_id, 'preferences'].iloc[0] != "":
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

        # Generate user preferences based on the top-rated movies with descriptions
        user_preferences = generate_preferences_from_rated_movies(rated_movies, model_name, chat_format, url, headers)
        
        # Update the preferences DataFrame with the new preferences for the current user
        preferences_df.loc[preferences_df['userId'] == user_id, 'preferences'] = user_preferences

        # Save the preferences to the CSV file every 100 users
        if user_id % 100 == 0:
            save_user_preferences(preferences_df, preferences_path)

    # Always save the preferences at the end
    save_user_preferences(preferences_df, preferences_path)

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

    # Test message for calling the API
    test_message = "Say 'this is a test.'"

    test_api_call(model_name, test_message, chat_format, api_url, headers)

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
        generate_all_user_preferences(ratings_dataframe, combined_dataframe, model_name, chat_format, preferences_path, api_url, headers, args.start_user_id)
        return

    # Ask how many users to add
    while True:
        try:
            num_users = int(input("How many users would you like to add? (Minimum 1): "))
            if num_users >= 1:
                break
            else:
                print("Please enter a number greater than or equal to 1.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

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

        print(f"\nPlease enter {n} movies you love and rate them so we can learn about what you like:")

        while len(new_ratings) < n:
            movie_title = input(f"Movie {len(new_ratings) + 1} title: ")
            
            # Get the IMDbId
            imdb_id = get_imdb_id_by_title(movie_title, model_name, chat_format, api_url, headers)
            if imdb_id:

                # Convert the found IMDbId to an integer
                imdb_id = int(imdb_id)

                # Convert IMDbId to MovieLens movieId
                movielens_id = imdb_to_movielens.get(imdb_id, None)
                if movielens_id:
                    # Map MovieLens MovieId back to title
                    confirmed_title = id_to_title.get(movielens_id, None)
                    if confirmed_title:
                        # Confirm with user    
                        confirmation = input(f"Is '{confirmed_title}' the movie you want to rate? (yes/no): ").strip().lower()
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
                                'timestamp': current_timestamp
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
    
        if describe_preferences == "no":

            # Prepare data for fetching descriptions
            rated_movies = [(rating['movieId'], id_to_title[rating['movieId']], rating['rating']) for rating in new_ratings]

            # Fetch movie descriptions for all newly rated movies
            movie_descriptions = get_movie_descriptions(rated_movies, combined_dataframe, model_name, chat_format, descriptions_path, api_url, headers, max_retries=5, delay_between_attempts=1)

            # Prepare data for preference generation
            rated_movies_with_descriptions = [
                (movie_id, title, movie_descriptions[movie_id], rating) for movie_id, title, rating in rated_movies
            ]

            # Generate preferences using the rated movies
            user_preferences = generate_preferences_from_rated_movies(rated_movies_with_descriptions, model_name, chat_format, api_url, headers)
        else:
            # Get user input for preferences
            user_preferences = input("Please describe what kind of movie experiences you are looking for (1 or 2 sentences):\n")

        # Output the generated or user-provided preferences
        print("\nUser Preferences:")
        print(user_preferences)

        # Create a temporary dataframe for the new preference
        new_preference = pd.DataFrame([{'userId': new_user_id, 'preferences': user_preferences}])

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

    cumulative_hit_rate_svd_10, cumulative_hit_rate_llm_10 = calculate_cumulative_hit_rate(
        algo, ratings_dataframe, id_to_title, combined_dataframe, model_name, chat_format,
        descriptions_path, api_url, headers, preferences_path, n=10, threshold=4.0, user_ids=new_user_ids, use_llm=True
    )

    '''
    cumulative_hit_rate_svd_100, cumulative_hit_rate_llm_100 = calculate_cumulative_hit_rate(
        algo, ratings_dataframe, id_to_title, combined_dataframe, model_name, chat_format,
        descriptions_path, api_url, headers, preferences_path, n=10, threshold=4.0, user_ids=new_user_ids, use_llm=True
    )
    '''

    print(f"\nCalculating Hit Rate for new users (SVD, threshold=4.0, n=10): {cumulative_hit_rate_svd_10:.2f}")
    print(f"Cumulative Hit Rate for new users (LLM-enhanced, threshold=4.0, n=10): {cumulative_hit_rate_llm_10:.2f}")

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

        # Get movie descriptions for the top N * 10 movies
        movie_descriptions = get_movie_descriptions(top_n_for_user_extended, combined_dataframe, model_name, chat_format,
                                                    descriptions_path, api_url, headers, max_retries=5, delay_between_attempts=1)

        # Get recommendations using LLM-enhanced method
        top_n_similar_movies = find_top_n_similar_movies(user_preferences, movie_descriptions, id_to_title, model_name, chat_format, n, api_url, headers)

        # Print the best movie recommendations according to the traditional algorithm enhanced by the LLM
        print(f"\nTop {n} recommendations according to LLM-enhanced method:")
        for movie_id, score in top_n_similar_movies:
            movie_title = id_to_title[movie_id]
            print(f"Movie Title: {movie_title}, Similarity Score: {score}")
    
if __name__ == "__main__":
    main()

# Test Code

# Print the updated ratings dataframe
# print(ratings_dataframe.tail(n))

'''
# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Evaluate SVD algorithm with cross validation
algo_to_evaluate = SVD()

# Train the algorithm on the trainset
print("Training SVD model on the trainset...\n")
algo_to_evaluate.fit(trainset)

# Evaluate SVD algorithm
evaluate_model(algo_to_evaluate, testset, data)

# Create a seperate SVD algorithm instance for calculating hit rate
algo_for_hit_rate = SVD()

# Create a seperate SVD algorithm instance for calculating cummulative hit rate
algo_for_cumulative_hit_rate = SVD()

# Calculate Normal Hit Rate with a threshold of 0 
print("Calculating Normal Hit Rate...\n")
normal_hit_rate = calculate_cumulative_hit_rate(algo_for_hit_rate, ratings_dataframe,  n=10, threshold=0)
print(f"Normal Hit Rate (threshold=0): {normal_hit_rate:.2f}")

# Calculate Cumulative Hit Rate with a threshold of 4.0
print("Calculating Cumulative Hit Rate...\n")
cumulative_hit_rate = calculate_cumulative_hit_rate(algo_for_cumulative_hit_rate, ratings_dataframe, n=10, threshold=4.0)
print(f"Cumulative Hit Rate (threshold=4.0): {cumulative_hit_rate:.2f}")
'''

# Code to test Cinemagoer API
'''
# Create an instance of the Cinemagoer class
ia = Cinemagoer()

movie_title = "Toy Story (1995)"
movie_description = ia_test_function(movie_title, combined_dataframe, model_name, chat_format, api_url, headers)
'''

# Code to test the API
'''
# Test message for calling the LLM API
test_message = "Say this is a test."

test_api_call(model_name, test_message, chat_format, api_url, headers)
'''