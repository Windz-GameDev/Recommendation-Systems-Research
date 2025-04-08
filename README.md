# LLM-enhanced Traditional Algorithms Recommendation System Framework

## Introduction

This GRE Research Experience project combines traditional recommendation algorithms with Large Language Models (LLMs) to provide personalized and context-aware movie recommendations. Optimized to work efficiently with compact models like Phi-4 running locally through KoboldCPP, the system demonstrates significant improvements in recommendation quality across multiple metrics compared to traditional SVD methods alone.

## Key Features

1. **Small Model Optimization**: Specifically designed to work efficiently with Phi-4 running locally through KoboldCPP with context shifting to reduce reprocessing.

2. **Hybrid Architecture**: Combines traditional collaborative filtering algorithms with LLM-based content understanding for improved recommendations.

3. **Multiple Algorithm Support**: Implements SVD, and SVD++ algorithms with LLM enhancement.

4. **Comprehensive Evaluation**: Provides detailed metrics including NDCG, MAP, Precision, Recall, Hit Rate, and Cumulative Hit Rate.

5. **Personalization**: Uses LLM-based similarity scoring between user preferences and movie descriptions.

6. **IMDb Integration**: Retrieves movie descriptions, ratings, and popularity scores to enhance recommendation quality.

7. **Scalable Design**: Works with both small datasets (MovieLens Latest Small) and large datasets (MovieLens 32M).

## Local LLM Optimization

- **KoboldCPP Integration**: Designed to work seamlessly with KoboldCPP for local LLM hosting
- **Context Shifting**: Prompts are sent and ordered in such a way that instead of processing approximately 1000-1500 tokens each time, only approximately 100 tokens are processed for the vast majority of prompts due to Kobold's Context Shifting.
- **Reduced Reprocessing**: Caches movie descriptions and user preferences to minimize redundant LLM calls
- **Few-shot Learning**: Uses carefully crafted few-shot examples to guide smaller models toward consistent similarity scoring with retry in case of failure and regex searching of responses to extract the numerical scores
- **Optimized Prompts**: Streamlined prompts that maximize performance on small models while minimizing token usage

## Recent Results

Our recent experiments with the MovieLens-Latest-Small dataset demonstrate significant improvements when using LLM enhancement:

### Ranking Metrics (K=10)

| Algorithm | NDCG@10   | MAP@10    | Precision@10 | Recall@10 |
| --------- | --------- | --------- | ------------ | --------- |
| SVD       | 0.090     | 0.042     | 0.078        | 0.026     |
| SVD LLM   | 0.157     | 0.080     | **0.130**    | **0.051** |
| SVD++     | 0.090     | 0.045     | 0.081        | 0.023     |
| SVD++ LLM | **0.160** | **0.085** | **0.130**    | 0.050     |

### Hit Rate Comparison

| Metric                      | SVD LLM      | SVD      | Improvement |
| --------------------------- | ------------ | -------- | ----------- |
| Hit Rate N@1                | **0.008197** | 0.001639 | 5.0x        |
| Hit Rate N@5                | **0.015847** | 0.006557 | 2.4x        |
| Hit Rate N@10               | **0.018579** | 0.009290 | 2.0x        |
| Cumulative Hit (≥ 4.0) N@1  | **0.011662** | 0.001944 | 6.0x        |
| Cumulative Hit (≥ 4.0) N@5  | **0.024295** | 0.008746 | 2.8x        |
| Cumulative Hit (≥ 4.0) N@10 | **0.029155** | 0.011662 | 2.5x        |

### Performance Averages

| Metric                      | SVD      | SVD LLM      | SVD++    | SVD++ LLM |
| --------------------------- | -------- | ------------ | -------- | --------- |
| Average Hit Rate            | 0.005829 | **0.014208** | 0.004372 | 0.009836  |
| Average Cumulative Hit Rate | 0.007451 | **0.021704** | 0.006575 | 0.017094  |

## How It Works

1. **Data Preparation**: The system loads movie ratings, metadata, and IMDb links.

2. **User Preference Input**: Users provide ratings and preferences which are used as input for both traditional algorithms and LLM enhancement.

3. **Base Recommendations**: Traditional algorithms (SVD and SVD++) generate initial recommendations.

4. **Movie Description Retrieval**: Descriptions are retrieved from IMDb or generated using LLMs when unavailable, then cached to avoid redundant processing.

5. **LLM-enhanced Similarity Scoring**: Using context-shifted prompts optimized for easy understanding by small models, LLMs generate similarity scores between user preferences and movie descriptions.

6. **Recommendation Refinement**: Base recommendations are refined using LLM-generated similarity scores.

7. **Comprehensive Evaluation**: The system calculates various metrics to quantify recommendation quality.

## Evaluation Methods

The system supports multiple evaluation approaches:

1. **Leave-One-Out Validation**: Calculates hit rates by removing one item per user.

2. **Stratified Train-Test Split with Ranking Metrics**: Uses 75/25 split to evaluate NDCG, MAP, Precision, Recall.

3. **Stratified Train-Test Split with Rating Metrics**: Uses 75/25 split to evaluate RMSE, MAE, R², Explained Variance. (Note: we don't focus on predicting the exact rating scores, but rather the best overall ordering of the recommendations)

4. **Combined Evaluation**: Performs both ranking and rating metrics evaluation.

5. **All of the Above**: Uses both Leave-One-Out Validation, and Stratified Train-Test Split metric evaluations.

## Modes of Operation

1. **Development Mode**: Works with a locally running LLM model (default):

   ```bash
   python llm_recommendation_system.py --mode development
   ```

2. **Production Mode**: Uses OpenAI API:

   ```bash
   python llm_recommendation_system.py --mode production
   ```

   Production mode requires you to create a .env file, and set your own OPENAI_API_KEY. We do not
   recommend using production mode due to high number of API calls made by the system and instead recommend
   using development mode for local execution.

3. **Generate Data Mode**: Prepopulates movie descriptions and user preferences:

   ```bash
   python llm_recommendation_system.py --mode generate-data --start-movie-id 1 --start-user-id 1
   ```

4. **Time Measurement Mode**: Evaluates recommendation generation speed:
   ```bash
   python llm_recommendation_system.py --mode time-measurement --sample-size 100 --sample-method first
   ```

## Getting Started

1. **Clone the repository**

2. **Set up Python environment**:

   ```bash
   conda install -c conda-forge scikit-surprise pandas numpy requests cinemagoer
   pip install -r requirements.txt
   ```

3. **Set up KoboldCPP with Phi-4**:

   - Download KoboldCPP from [GitHub](https://github.com/LostRuins/koboldcpp)
   - Download Phi-4 model (recommended: phi-4-Q6_K.gguf) from Hugging Face
   - Run KoboldCPP with the Phi-4 model with OpenAI API compatibility on port 5001
   - Important: Ensure that 'Use ContextShift' is checked

4. **Prepare the dataset**:

   - Create `Datasets/Movie_Lens_Datasets/` directories if not already present
   - Download MovieLens dataset (small or 32M version)
   - Extract contents to the appropriate folder:
     - Small: `ml-latest-small/`
     - 32M: `ml-32m/`

5. **Run the system**:

   ```bash
   python llm_recommendation_system.py
   ```

6. **Interactive Usage**:
   - Add new users with ratings
   - Generate personalized recommendations
   - Calculate evaluation metrics

## Advanced Features

- **Context-Optimized Prompts**: Special prompt formatting optimized for Phi-4 and similar models
- **User Preference Generation**: LLM-based generation of preference summaries from rated movies
- **Date Range Preference**: Automatic detection of preferred movie release date ranges
- **Rating/Popularity Preferences**: Detection of user preferences for highly-rated or popular movies
- **Performance Measurement**: Detailed timing of algorithmic components

## Future Directions

1. **Further Prompt Optimization**: Continuing to refine prompting strategies for smaller models
2. **Memory-Efficient Processing**: Implementing additional optimization techniques to reduce memory footprint and improve program efficiency
3. **Enhanced User Preference Understanding**: Improving extraction of user preferences
4. **Further Algorithm Integration**: Expanding support for state-of-the-art recommendation approaches
5. **Improved Similarity Score Generation**: Enhancing the accuracy of LLM-generated similarity scores by incorporating more user and movie data in score calculation

## License

This project is licensed under the MIT License.
