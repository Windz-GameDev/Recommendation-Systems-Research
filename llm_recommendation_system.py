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

# Function to get top N recommendations for a user
def get_top_n_recommendations(algo, user_id, all_movie_ids, ratings_dataframe, n=10):
    """
    Get top N movie recommendations for a user.

    Parameters:
    - algo: The trained recommendation algorithm.
    - user_id: The ID of the user for whom to generate recommendations.
    - all_movie_ids: List of all movie IDs in the dataset.
    - ratings_dataframe: The dataframe used to generate predictions. 
    The testset should be removed if this function is being called for hitrate.
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

def calculate_cumulative_hit_rate(algo, ratings_dataframe, n=10, threshold=4.0):
    '''
    Calculate the cumulative hit rate of the recommendation algorithm using leave-one-out cross validation.

    Parameters:
    - algo: The Trained recommendation algorithm
    - ratings_dataframe: The DataFrame containing all movie ratings made by users.
    - n: The number of top recommendations to consider.
    - threshold: The rating threshold to consider an item as relevant. Passing 0 essentially causes this function to behave as normal hit rate.
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
            top_n_recommendations = get_top_n_recommendations(algo, user_id, all_movie_ids, ratings_dataframe_testset_removed, n)

            # Check if the left-out movie is in the top N recommendations
            if movie_id in [rec_movie_id for rec_movie_id, _ in top_n_recommendations]:
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
    
def get_imdb_id_by_title(title):
    '''
    Retrieve the IMDb ID for a movie given its title.

    Parameters:
    - title (str): The title of the movie.

    Returns:
    - str: The IMDb ID of the movie, or None if not found.
    '''

    ia = Cinemagoer()
    try:
        # Search for the movie by title
        search_results = ia.search_movie(title)
        for movie in search_results:
         # Check for an exact title match with case sensitivity
            if movie['title'] == title:
                imdb_id = movie.movieID
                print(f"Found exact match for '{title}': IMDb ID is {imdb_id}")
                return imdb_id
            print(f"No exact match found for title: {title}")
            return None
        else:
            print(f"No results found for title: {title}")
            return None
    except IMDbError as e:
        print(f"An error occurred while searching for movie {title}'s IMDb ID: {e}")
        return None

def get_movie_with_retries(imdb_id, movie_title, max_retries=10, delay=1):
    
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
                print(f"The movie data for {movie_title} was retrieved successfully.")
                return movie
        except IMDbError as e:
            # Check if the error is an HTTP 404 error
            if 'HTTPError 404' in str(e):
                print(f"HTTP 404 error encountered for {movie_title}'s IMDd ID {imdb_id}.  Attempting to find IMDb ID by title.")
                imdb_id = get_imdb_id_by_title(movie_title)
                if not imdb_id:
                    print(f"Could not find the IMDd ID with the title '{movie_title}'.")
                    return None
            else:
                print(f"Attempt {attempt + 1} failed: {e}")

        attempt += 1
        if attempt < max_retries:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"Max retries reached. Could not fetch the plot for {movie_title}.")
            return None

def get_movie_descriptions(top_n_movies, combined_dataframe, model_name, selected_description_chat_format, max_retries, delay_between_attempts):

    """
    Retrieves movie descriptions for a list of top N movies using their IMDb IDs.

    Parameters:
    - top_n_movies (list): A list of tuples containing movie IDs and their estimated ratings.
    - combined_dataframe (DataFrame): A DataFrame containing movie information, including IMDb IDs.

    Returns:
    - descriptions (dict): A dictionary mapping movie IDs to their descriptions.
    """
    descriptions = {}
    for movie_id, movie_title, _ in top_n_movies:
        imdb_id = combined_dataframe.loc[combined_dataframe['movieId'] == movie_id, 'imdbId'].values[0]
        movie = get_movie_with_retries(imdb_id, movie_title, max_retries, delay_between_attempts)
        if movie and 'plot' in movie:
            descriptions[movie_id] = movie['plot'][0] if isinstance(movie['plot'], list) else movie['plot']
        else:
            # Could not find a description, as a last resort, generate a description using few shot prompting
            descriptions[movie_id] = generate_description_with_few_shot(movie_title, model_name, selected_description_chat_format)
            print(descriptions[movie_id])
            # descriptions[movie_id] = f"No description available for {movie_title}."
    return descriptions


def find_most_similar_movie(user_input, movie_descriptions, model_name, chat_format, url="http://localhost:5001/v1/chat/completions"):
   
    """
    Finds the most similar movie to the user's input from a list of movie descriptions using an LLM.

    Parameters:
    - user_input (str): The user's input describing their preferences.
    - movie_descriptions (dict): A dictionary mapping movie IDs to their descriptions.
    - model_name (str): The name of the model to use for generating the similarity score.
    - chat_format (str): A format string to structure the user input and movie description.
    - url (str): The API endpoint URL to send the request to. Defaults to "http://localhost:5001/v1/chat/completions".

    Returns:
    - best_match (int): The movie ID of the most similar movie, or None if no match is found.
    """

    headers = {"Content-Type": "application/json"}
    best_match = None
    highest_similarity = -1

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
            
            if similarity_score > highest_similarity:
                highest_similarity= similarity_score
                best_match = movie_id
        
    return best_match

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

# Create id to title and title to id mappings
id_to_title, title_to_id = create_movie_mappings(movies_dataframe)

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
normal_hit_rate = calculate_cumulative_hit_rate(algo_for_hit_rate, ratings_dataframe, n=10, threshold=0)
print(f"Normal Hit Rate (threshold=0): {normal_hit_rate:.2f}")

# Calculate Cumulative Hit Rate with a threshold of 4.0
print("Calculating Cumulative Hit Rate...\n")
cumulative_hit_rate = calculate_cumulative_hit_rate(algo_for_cumulative_hit_rate, ratings_dataframe, n=10, threshold=4.0)
print(f"Cumulative Hit Rate (threshold=4.0): {cumulative_hit_rate:.2f}")
'''

# Load the links dataset into a pandas dataframe
links_dataframe = pd.read_csv(links_path)
# print(links_dataframe.head())

# Merge the dataframes on 'movieId'
combined_dataframe = pd.merge(movies_dataframe, links_dataframe, on='movieId')
# print(combined_dataframe.head())

# Code to test Cinemagoer API
'''
# Create an instance of the Cinemagoer class
ia = Cinemagoer()

movie_title = "Toy Story (1995)"
movie_description = ia_test_function(movie_title, combined_dataframe)
'''

# Initialize data for connecting to API

# Define chat format for Phi-3-mini-4k-instruct-gguf
phi_chat_format = "<|user|>\n{prompt} <|end|>\n<|assistant|>"

# Modify to use existing chat model formats when switching between models as need
selected_test_chat_format = phi_chat_format

# Put model name you want to here for the test API call
test_model_name = "Phi-3-mini-4k-instruct-q4.gguf"

'''
# Test message for calling the API
test_message = "Say this is a test."

test_api_call(test_model_name, test_message, selected_test_chat_format)
'''

# Define user message
user_input = input("Please describe your movie preferences: ")

# Modify depending on the model
selected_chat_format = phi_chat_format

model_name = "Phi-3-mini-4k-instruct-q4.gguf"

# Get a list of all movie IDs
all_movie_ids = movies_dataframe['movieId'].unique()

# Create an algorithm for providing the user a recommendation
algo_for_user = SVD()

# Build the trainset from the full dataset
full_trainset = data.build_full_trainset()

# Train the algorithm using the full trainset
algo_for_user.fit(full_trainset)

# Specify the user ID for which to generate recommendations
specific_user_id = 1

top_n_for_user = get_top_n_recommendations(algo_for_user, specific_user_id, all_movie_ids, ratings_dataframe, n=100)

# Print the top N recommendations for the specific user with estimated ratings
print(f"Top 10 recommendations for User {specific_user_id}:")
for movie_id, movie_title, est_rating in top_n_for_user:
    print(f"Title: {movie_title}, Estimated Rating: {est_rating:.2f}")

# Get movie descriptions
movie_descriptions = get_movie_descriptions(top_n_for_user, combined_dataframe, model_name, selected_chat_format, max_retries=5, delay_between_attempts=0.01)

# Find the most similar movie
most_similar_movie_id = find_most_similar_movie(user_input, movie_descriptions, model_name, selected_chat_format)

# Print the most similar movie
if most_similar_movie_id:
    print(f"The movie most similar to your interests is: {id_to_title[most_similar_movie_id]}")
else:
    print("No similar movie found.")
    
