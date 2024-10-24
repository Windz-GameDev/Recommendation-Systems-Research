# LLM-Enhanced Recommendation System Framework for GRE Research Project

## Project Overview

This repository contains the code and resources for a research project focused on developing a versatile recommendation system framework enhanced by Large Language Models (LLMs). The project aims to explore the integration of LLMs to improve the accuracy and personalization of recommendations. This work is part of a GRE research project.

## Features

- **Versatile Recommendation Framework**: A flexible system that can be adapted to various recommendation scenarios.
- **LLM Integration**: Enhances traditional recommendation algorithms with LLM capabilities for improved personalization.
- **Cumulative Hit Rate Calculation**: Implements leave-one-out cross-validation to calculate the cumulative hit rate of the recommendation algorithm.
- **Movie Description Generation**: Uses few-shot prompting with a language model to generate movie descriptions when not available.
- **Similarity Matching**: Finds the most similar movie to user preferences using an LLM.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Windz-GameDev/Recommendation-Systems-Research cd <repository-directory>
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Ensure you have the necessary datasets in the `Datasets` directory.

## Usage

1. **Train and Evaluate the Model**:

   - The script trains an SVD model on the MovieLens dataset and evaluates its performance using RMSE and MAE metrics.

2. **Generate Recommendations**:

   - Run the script to generate top N movie recommendations for a specific user.

3. **Calculate Hit Rate**:

   - The script calculates both normal and cumulative hit rates to evaluate the recommendation algorithm's effectiveness.

4. **Generate Movie Descriptions**:

   - The script retrieves or generates movie descriptions using IMDb data and few-shot prompting.

5. **Find Similar Movies**:
   - Input your movie preferences, and the script will find the most similar movie using an LLM.

## Methodology

- **Large Language Models (LLMs)**: The project leverages LLM techniques to enhance the recommendation system's ability to understand and predict user preferences.
- **Surprise Library**: Utilizes the Surprise library for collaborative filtering and recommendation algorithm implementation.
- **Cinemagoer API**: Integrates with the IMDb database to fetch movie information and descriptions.

## LLM API Integration

This application is designed to work with a locally running LLM model that exposes an OpenAPI-compatible API on port 5001. An example application for running local LLMs is [KoboldCPP](https://github.com/LostRuins/koboldcpp), which allows you to run language models on your local machine. Ensure that the LLM model is running and accessible at `http://localhost:5001/v1/chat/completions` before executing llm_recommendation_system.py. 

## Datasets

- **MovieLens**: The project uses the MovieLens dataset for training and evaluating the recommendation system.
- **IMDb**: The IMDb dataset is used to retrieve movie descriptions and additional metadata.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
