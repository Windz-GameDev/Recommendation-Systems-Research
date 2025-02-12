### Test Functions for the Movie Recommendation System Not Currently in Use

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

def get_movie_scores(
    movie_id: Union[int, str],
    imdb_id: Union[int, str],
    movie_title: str
) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Fetches the IMDb rating and calculates a normalized popularity score for a movie/TV show.

    This function uses the Cinemagoer library to retrieve movie details using the provided IMDb ID.
    It then extracts two main pieces of information:
      - The IMDb rating (a float, if available).
      - The number of votes cast on IMDb (an integer, if available).

    Using the raw vote count, it computes a normalized popularity score on a 0–100 scale using the 
    'normalize_popularity_score' function. This normalized score helps gauge popularity in a standard format.

    Parameters:
      movie_id (int or str):
          Internal movie identifier for context (e.g., MovieLens ID). It is not directly used for the lookup.
      imdb_id (int or str):
          The IMDb identifier for the movie or TV show, which is used to query the IMDb database via Cinemagoer.
      movie_title (str):
          The title of the movie. This is provided for logging purposes in case data retrieval fails.

    Returns:
      tuple:
          A tuple containing three elements:
            - rating (Optional[float]): The IMDb rating (e.g., 7.8) if available; otherwise, None.
            - normalized_popularity (Optional[int]): A popularity score scaled between 0 and 100 derived 
              from the raw vote count; None if vote data is unavailable.
            - votes (Optional[int]): The raw number of votes from IMDb; None if unavailable.

    Functionality:
      1. A Cinemagoer instance is created to access IMDb data.
      2. The function calls ia.get_movie(imdb_id) to fetch the movie details.
      3. If the movie exists:
          - The IMDb rating is retrieved with movie.get('rating', None).
          - The total vote count is retrieved with movie.get('votes', None).
          - If votes are available, the function calls normalize_popularity_score(votes) to compute
            a normalized popularity score on the range 0–100.
      4. If any IMDbError is encountered (e.g., due to an invalid IMDb ID or connectivity issues),
         an error message is printed using the movie_title for clarification, and the function returns (None, None, None).
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

def evaluate_model(algo: Any, testset: List[Any], data: Any) -> None:
    """
    Evaluate the performance of a recommendation algorithm using standard metrics and cross-validation.

    This function uses a trained recommendation algorithm (typically from the Surprise library)
    to predict ratings for a provided test set. It then calculates two key evaluation metrics:
      - RMSE (Root Mean Squared Error): Measures the average magnitude of the error by 
        taking the square root of the average squared differences between predicted and actual ratings.
        RMSE is sensitive to larger errors.
      - MAE (Mean Absolute Error): Measures the average absolute difference between
        predicted and actual ratings.

    After evaluating the model on the test set, the function also performs 5-fold cross-validation
    on the full dataset to provide additional insight into the model's generalization performance.

    Parameters:
      algo:
        The trained recommendation algorithm. This object must implement a .test() method (as provided by Surprise)
        and be compatible with cross_validate (e.g., SVD, KNNBasic, etc.).
        
      testset:
        A list of (user, item, actual rating) tuples representing the test set.
        
      data:
        The full dataset object (e.g., a Surprise Dataset) used for cross-validation.
        This should include necessary information (e.g., user IDs, item IDs, and ratings).

    Returns:
      None

    Functionality:
      1. Prints a message indicating the start of evaluation.
      2. Uses the algo.test() method to generate predictions for all entries in the test set.
      3. Computes and prints RMSE and MAE metrics based on the predictions.
      4. Executes a 5-fold cross-validation on the full dataset and prints the detailed metrics
         (RMSE, MAE for each fold) as provided by the cross_validate function.
    """
    print(f"Evaluating {algo} algorithm on the test set...")

    # Get predictions for the test set
    predictions = algo.test(testset)

    # Compute and print RMSE
    rmse = accuracy.rmse(predictions, verbose=True)
    print("RMSE (Root Mean Squared Error) measures the differences between predicted and actual ratings. "
          "It gives higher weight to larger errors, making it sensitive to outliers.")

    # Compute and print MAE
    mae = accuracy.mae(predictions, verbose=True)
    print("MAE (Mean Absolute Error) is the average of the absolute differences between predicted and actual ratings. "
          "It provides a straightforward measure of prediction accuracy.")

    # Perform cross-validation and display average performance metrics for 5 folds
    print("Performing 5-fold cross-validation on the full dataset...")
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Test Code Not In Use

# Print the updated ratings dataframe
# print(ratings_dataframe.tail(n))

'''
# Test message for calling the API
test_message = "Say 'this is a test.'"

test_api_call(model_name, test_message, chat_format, api_url, headers)

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