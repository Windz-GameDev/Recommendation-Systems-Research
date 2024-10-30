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

def calculate_cumulative_hit_rate(algo, ratings_dataframe, id_to_title, n=10, threshold=4.0):
    '''
    Calculate the cumulative hit rate of the recommendation algorithm using leave-one-out cross-validation.

    This function evaluates the performance of a recommendation algorithm by determining how often a relevant item (as defined by the threshold) is present in the top N recommendations for each user. It uses a leave-one-out cross-validation approach, where one rating is removed from each user's data to form a test set, and the algorithm is trained on the remaining data.

    Parameters:
    - algo: The trained recommendation algorithm. This is an instance of a recommendation algorithm from the Surprise library, such as SVD.
    - ratings_dataframe: A pandas DataFrame containing all movie ratings made by users. It should have columns 'userId', 'movieId', and 'rating'.
    - id_to_title: A dictionary mapping movieId to movie title. This is used to convert movie IDs to human-readable titles in the recommendations.
    - n: The number of top recommendations to consider for each user. Default is 10.
    - threshold: The rating threshold to consider an item as relevant. A rating equal to or above this threshold is considered relevant. Passing 0 causes this function to behave as a normal hit rate calculation, where any rating is considered relevant.

    Returns:
    - hit_rate: The cumulative hit rate, which is the proportion of relevant items that appear in the top N recommendations across all users.
    '''

    # Create a copy of the ratings DataFrame which will have the test set ratings removed
    ratings_dataframe_testset_removed = ratings_dataframe.copy()

    # Initialize the leave-one-out test set
    loo_testset = []

    # Get unique user IDs from the ratings dataset
    unique_user_ids = ratings_dataframe_testset_removed['userId'].unique()

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
    hit_count = 0
    total_count = 0

    # Iterate over each user in the leave-one-out test set
    for user_id, movie_id, rating in loo_testset:

        # Check if the left-out rating is about the threshold
        if rating >= threshold:
            total_count += 1

            # Get the top N recommendations for the user, passing the dataset with the test set removed
            top_n_recommendations = get_top_n_recommendations(algo, user_id, all_movie_ids, ratings_dataframe_testset_removed, id_to_title, n)

            # Check if the left-out movie is in the top N recommendations
            if movie_id in [rec_movie_id for rec_movie_id, _, _ in top_n_recommendations]:
                hit_count += 1
            
    # Calculate hit rate
    hit_rate = hit_count / total_count if total_count > 0 else 0

    return hit_rate 

def generate_description_with_few_shot(movie_title, model_name, selected_description_chat_format, url="http://localhost:5001/v1/chat/completions"):
    """
    Generate a movie description using few-shot prompting with a language model.
    
    Parameters:
    - movie_title (str): The title of the movie.
    - model_name (str): The name of the model to use for generating the description.
    - selected_description_chat_format (str): The format string to structure the few-shot examples and movie title.
    - url (str): The API endpoint URL to send the request to.

    Returns: 
    = str: The generated movie description.
    """

    headers = {"Content-Type": "application/json"}

    # Few-shot examples
    few_shot_examples = (
        "Example 1:\n"
        "Movie title: Inception\n"
        "Description: A mind-bending thriller where a skilled thief is given a chance at redemption if he can successfully perform an inception.\n\n"
        "Example 2:\n"
        "Movie title: The Matrix\n"
        "Description: A computer hacker learns about the true nature of reality and his role in the war against its controllers.\n\n"
    )

    # Format the prompt using the provided chat format 
    prompt = selected_description_chat_format.format(prompt=few_shot_examples + "Now, generate a description for the following movie:\n" + f"Movie title: {movie_title}\nDescription:")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates movie descriptions."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 120,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
    else:
        return "No description could be found or generated."
    
def get_imdb_id_by_title(title, model_name, chat_format, url="http://localhost:5001/v1/chat/completions"):
    '''
    Retrieve the IMDb ID for a movie given its title, using LLM for fuzzy matching if necessary.

    Parameters:
    - title (str): The title of the movie.
    - model_name (str): The name of the model to use for generating the similarity score.
    - chat_format (str): A format string to structure the user input and movie description
    - url (str): The API endpoint URL to send the request to.

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
        headers = {"Content-Type": "application/json"}
        best_match = None
        highest_similarity = -1  

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

def get_movie_with_retries(imdb_id, movie_title, model_name, chat_format, max_retries=10, delay=1):
    
    """
    Attempts to retrieve a movie object with plot and reviews from IMDb, retrying if necessary.

    Parameters:
    - imdb_id (str): The IMDb ID of the movie to retrieve.
    - max_retries (int): The maximum number of retry attempts. Defaults to 10.
    - delay (int): The delay in seconds between retry attempts. Defaults to 1.

    Returns:
    - movie (dict): The movie object containing plot and reviews if successful, otherwise None.
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
                imdb_id = get_imdb_id_by_title(movie_title, model_name, chat_format)
                if not imdb_id:
                    print(f"Could not find the IMDd ID with the title '{movie_title}'.")
                    return None
            else:
                print(f"Attempt {attempt + 1} failed: {e}")

        attempt += 1
        if attempt < max_retries:
            # print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            #print(f"Max retries reached. Could not fetch the plot for {movie_title}.")
            return None

def get_movie_descriptions(top_n_movies, combined_dataframe, model_name, selected_chat_format, max_retries, delay_between_attempts):

    """
    Retrieves movie descriptions for a list of top N movies using their IMDb IDs.

    Parameters:
    - top_n_movies (list): A list of tuples containing movie IDs and their estimated ratings.
    - combined_dataframe (DataFrame): A DataFrame containing movie information, including IMDb IDs.

    Returns:
    - descriptions (dict): A dictionary mapping movie IDs to their descriptions.
    """
    descriptions = {}
    for index, (movie_id, movie_title, _) in enumerate(top_n_movies):
        if index % 10 == 0:
            print("Retrieving movie descriptions...")

        imdb_id = combined_dataframe.loc[combined_dataframe['movieId'] == movie_id, 'imdbId'].values[0]
        movie = get_movie_with_retries(imdb_id, movie_title, model_name, selected_chat_format, max_retries, delay_between_attempts)
        if movie and 'plot' in movie:
            descriptions[movie_id] = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
        else:
            # Could not find a description, as a last resort, generate a description using few shot prompting
            descriptions[movie_id] = generate_description_with_few_shot(movie_title, model_name, selected_chat_format)
            # print(descriptions[movie_id])
            # descriptions[movie_id] = f"No description available for {movie_title}."
    return descriptions


def find_top_n_similar_movies(user_input, movie_descriptions, id_to_title, model_name, chat_format, n, url="http://localhost:5001/v1/chat/completions"):
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
        - url (str): The API endpoint URL to send the request to. Defaults to "http://localhost:5001/v1/chat/completions".

        Returns:
        - top_n_movies (list of tuples): A list of tuples containing the movie ID and similarity score of the top N most similar movies.
    """

    headers = {"Content-Type": "application/json"}
    similarity_scores = []

    for movie_id, description in movie_descriptions.items():

        # Get the title for the prompt
        movie_title = id_to_title[movie_id]

        # Add example input and output to guide the LLM
        few_shot_examples = (
            "Example 1:\n"
            "User input: I love science fiction with deep philosophical themes.\n"
            "Movie title: Inception\n"
            "Movie description: A futuristic tale exploring the nature of consciousness and identity.\n"
            "Rate how likely you think the movie aligns with the user's interests (respond with a number):\n"
            "0.9\n"
            "Example 2:\n"
            "User input: I enjoy light-hearted comedies with a lot of humor.\n"
            "Movie title: The Dark Knight\n"
            "Movie description: A dark and intense drama about the struggles of life.\n"
            "Rate how likely you think the movie aligns with the user's interests (respond with a number):\n"
            "-0.7\n"
            "Example 3:\n"
            "User input: I am fascinated by historical documentaries.\n"
            "Movie title: The Lord of the Rings\n"
            "Movie description: An epic adventure set in a fantasy world with dragons and magic.\n"
            "Rate how likely you think the movie aligns with the user's interests (respond with a number):\n"
            "-0.5\n"
        )

        # Format the prompt using the provided chat format
        prompt_content = (
            f"User input: {user_input}\n"
            f"Movie title: {movie_title}\n"
            f"Movie description: {description}\n"
            "Rate how likely you think the movie aligns with the user's interests (respond with a number):\n"
        )

        full_prompt = few_shot_examples + "Now, respond to the following prompt:\n" + prompt_content
        
        # Format the prompt using the provided chat format
        full_prompt = chat_format.format(prompt=full_prompt)
        
        # Add example input and output to guide the LLM

        payload = {
            "model": model_name, # Use passed model name
            "messages": [
                {"role": "system", "content": "You are a helpful assistant whose job is to rate the similarity between user preferences and product descriptions. Always respond with a number between -1.0 and 1.0."},
                {"role":"user", "content": full_prompt}
            ],
            "max_tokens": 5,
            "temperature": 0
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            content = response.json().get("choices", [])[0].get("message", {}).get("content", "0")
            # print(content)
            similarity_score = float(content)
            similarity_scores.append((movie_id, similarity_score))

    # Sort the movies by similarity score in descending order and select the top N
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_n_movies = similarity_scores[:n]
 
    return top_n_movies

def ia_test_function(movie_title, combined_dataframe):
    """
    Tests the retrieval of movie information from IMDb for a given movie title.

    Parameters:
    - movie_title (str): The title of the movie to retrieve information for.
    - combined_dataframe (DataFrame): A DataFrame containing movie information, including IMDb IDs.

    Returns:
    - movie_description (str): The plot description of the movie, or a default message if not available.
    """
    # Access IMDb ID for a specific movie
    imdb_id = combined_dataframe.loc[combined_dataframe['title'] == movie_title, 'imdbId'].values[0]
    print(f"IMDb ID for '{movie_title}' is {imdb_id}")

    # Get the movie object with title, plot, and reviews
    movie = get_movie_with_retries(imdb_id, movie_title)

    # Access the plot (description)
    movie_description = None
    if movie:
        if isinstance(movie['plot'], list):
            movie_description = movie['plot'][0]  # We are using the first summary for simplicity
            print(f"Movie Description: {movie_description}")
        else:
            movie_description = movie['plot']
    else:
        movie_description = "No plot information available."

    # Add the description to the DataFrame
    combined_dataframe.loc[combined_dataframe['title'] == movie_title, 'description'] = movie_description

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

    return movie_description

def test_api_call(model_name, user_message, chat_format, url="http://localhost:5001/v1/chat/completions"):
    """
    Sends a POST request to a specified LLM API endpoint to generate a response based on a user message.

    Parameters:
    - model_name (str): The name of the model to use for generating the response.
    - user_message (str): The message or prompt from the user that needs a response.
    - chat_format (str): A format string to structure the user message. It should contain a placeholder for the user message.
    - url (str): The API endpoint URL to send the request to. Defaults to "http://localhost:5001/v1/chat/completions".

    Returns:
    - str: The generated response from the LLM if successful, otherwise None.
    """

    # Define the headers, for now omitting the API key
    headers = {
        "Content-Type": "application/json"
    }

    # Format user message using provided chat format
    user_prompt = chat_format.format(user_message=user_message)

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

def main():

    # Define MovieLens file paths

    # Contains userId, movieId, rating, timestamp
    ratings_path = 'Datasets/Movie_Lens_Datasets/ml-latest-small/ratings.csv'

    # Contains movieId, title, genres
    movies_path = 'Datasets/Movie_Lens_Datasets/ml-latest-small/movies.csv'

    # Contains movieId, imdbId, tmdbId. Essentially this serves as a mapping from MovieLens's movieID to Internet Movie Database's and The Movie Database's movie ids.
    links_path = 'Datasets/Movie_Lens_Datasets/ml-latest-small/links.csv'

    # Contains userId, movieId, tag, timestamp. Allows us to see what keywords and phrases users associated with different movies. 
    # This can allow us to better understand the content of movies when analyzing user preferences.
    tags_path = 'Datasets/Movie_Lens_Datasets/ml-latest-small/tags.csv'

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

    # Create id to title and title to id mappings
    id_to_title, title_to_id = create_movie_mappings(movies_dataframe)

    # Create IMDB id to MovieLens ID mapping and vice versa
    movielens_to_imdb, imdb_to_movielens = create_id_mappings(links_dataframe)

    # Number of movies the user wants to rate
    n = 5  # You can change this to any number you prefer

    # Assign a new user ID
    new_user_id = ratings_dataframe['userId'].max() + 1

    # Initialize data for connecting to LLM API

    # Define chat format for Phi-3-mini-4k-instruct-gguf
    phi_chat_format = "<|user|>\n{prompt} <|end|>\n<|assistant|>"

    # Modify depending on the model``
    selected_chat_format = phi_chat_format

    model_name = "Phi-3-mini-4k-instruct-q4.gguf"

    # Step 1: Ask user for 5 movies they love
    new_ratings = []

    print(f"Please enter {n} movies you love and rate them so we can learn about what you like:")

    while len(new_ratings) < n:
        movie_title = input(f"Movie {len(new_ratings) + 1} title: ")
        
        # Get the IMDbId
        imdb_id = get_imdb_id_by_title(movie_title, model_name, selected_chat_format)
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
                print(f"Movie '{movie_title}' does not exist in the MovieLens dataset of the current size. Please try another movie.")
        else:
            print(f"Could not find IMDbId for movie '{movie_title}'. Please try another movie.")

    # Create a dataframe for the new ratings using the list of dictionaries
    new_ratings_df = pd.DataFrame(new_ratings)

    # Concatenate the new ratings DataFrame with the existing ratings DataFrame
    ratings_dataframe = pd.concat([ratings_dataframe, new_ratings_df], ignore_index=True)

    # Print the updated ratings dataframe
    # print(ratings_dataframe.tail(n))

    # Add the new user's ratings to the dataframe

    # Define a Reader with the appropriate rating scale
    reader = Reader(rating_scale=(0.5, 5.0))

    # The dataframe must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings in this order. 
    # The reader object is also required with the rating_scale parameter specified.
    data = Dataset.load_from_df(ratings_dataframe[['userId', 'movieId', 'rating']], reader)

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
    movie_description = ia_test_function(movie_title, combined_dataframe)
    '''

    '''
    # Put model name you want to here for the test API call
    test_model_name = "Phi-3-mini-4k-instruct-q4.gguf"

    # Modify to use existing chat model formats when switching between models as need
    selected_test_chat_format = phi_chat_format

    # Test message for calling the API
    test_message = "Say this is a test."

    test_api_call(test_model_name, test_message, selected_test_chat_format)
    '''

    # Define user message
    user_input = input("Please describe what kind of movie experiences you are looking for (1 or 2 sentences):\n")

    # Get a list of all movie IDs
    all_movie_ids = movies_dataframe['movieId'].unique()

    # Create an algorithm for providing the user a recommendation
    algo_for_user = SVD()

    # Build the trainset from the full dataset
    full_trainset = data.build_full_trainset()

    # Train the algorithm using the full trainset
    algo_for_user.fit(full_trainset)

    # Specify the user ID for which to generate recommendations
    specific_user_id = new_user_id

    # Set number of recommendations to generate with SVD, we will pick the top ten to compare with LLM
    n_times_10 = 100

    if n_times_10 >= 10:
        n = int(n_times_10 / 10)
    else:
        n = n_times_10

    top_n_for_user = get_top_n_recommendations(algo_for_user, specific_user_id, all_movie_ids, ratings_dataframe, id_to_title, n_times_10)

    print("\n")

    # Print the top N recommendations for the specific user with estimated ratings
    print(f"Top {n} recommendations for User {specific_user_id} according to Traditional Algorithm:")
    for movie_id, movie_title, est_rating in top_n_for_user[:n]:
        print(f"Movie Title: {movie_title}, Estimated Rating: {est_rating:.2f}")

    print("\n")

    # Get movie descriptions
    movie_descriptions = get_movie_descriptions(top_n_for_user, combined_dataframe, model_name, selected_chat_format, max_retries=5, delay_between_attempts=0.01)

    print("\n")

    print(f"Top {n} recommendations for User {specific_user_id} according to Traditional Algorithm Enhanced by LLM:")
    # Print the best movie recommendations according to the traditional algorithm enhanced by the LLM
    top_n_similar_movies = find_top_n_similar_movies(user_input, movie_descriptions, id_to_title, model_name, selected_chat_format, n)
    for movie_id, score in top_n_similar_movies:
        print(f"Movie Title: {id_to_title[movie_id]}, Similarity Score (Matches your interests): {score}")
    
if __name__ == "__main__":
    main()
