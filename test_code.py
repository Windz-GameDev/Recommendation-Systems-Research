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


# Test Code

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