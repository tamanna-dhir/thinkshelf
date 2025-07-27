# ThinkShelf 📚

A hybrid book recommendation system that combines content-based filtering with collaborative filtering to suggest personalized book recommendations. Features both a command-line interface and a web-based chatbot powered by IBM Watson Assistant.

## Features

- **Hybrid Recommendation Engine**: Combines multiple recommendation approaches for better accuracy
- **Interactive Web Interface**: Gradio-powered UI with chatbot functionality
- **Command Line Tool**: Simple CLI for quick book recommendations
- **IBM Watson Integration**: Natural language processing for conversational interactions
- **Multiple Search Methods**: Search by title, author, genre, or partial matches
- **Visual Interface**: Book cover images and styled UI

## Project Structure

```
thinkshelf/
├── book_recommender.py      # Main web application with Gradio UI
├── thinkshelf_export.py     # Command-line interface
├── test.py                  # Contains hybrid_recommend_clean function
├── models/                  # Machine learning models and data
│   ├── books_cleaned.pkl    # Cleaned book dataset
│   ├── indices.pkl          # Book title indices
│   ├── nn_model.pkl         # Nearest neighbors model
│   ├── tfidf_matrix.pkl     # TF-IDF vectorized features
│   ├── similarity_scores.pkl # Precomputed similarity scores
│   └── pt.pkl               # Pivot table for collaborative filtering  
└── static/                  # UI assets
    ├── background.jpg       # Background image
    └── dobby.png           # Chatbot avatar
```

## Requirements

```bash
pip install pandas numpy scikit-learn gradio ibm-watson ibm-cloud-sdk-core
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd thinkshelf
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up model files**
   - Ensure all `.pkl` files are in the `models/` directory
   - Add background image and Dobby avatar to `static/` directory

4. **Configure IBM Watson** (for web interface)
   - Update API credentials in `book_recommender.py`:
     ```python
     API_KEY = "your-watson-api-key"
     ASSISTANT_ID = "your-assistant-id" 
     URL = "your-watson-service-url"
     ```

## Usage

### Command Line Interface

Run the CLI version for quick recommendations:

```bash
python thinkshelf_export.py
```

Example interaction:
```
📚 Welcome to ThinkShelf Book Recommender!
🔎 Enter a book title, author, or genre: Harry Potter

📖 Top Recommendations:
1. Title: The Hobbit
   Author: J.R.R. Tolkien
   Genre: Fantasy
```

### Web Interface

Launch the Gradio web application:

```bash
python book_recommender.py
```

This will start a web server (typically at `http://localhost:7860`) where you can:
- Chat with Dobby for book recommendations
- Get personalized suggestions based on natural language queries
- View book covers and detailed information

## How It Works

### Recommendation Algorithm

The system uses a **hybrid approach** combining:

1. **Content-Based Filtering**
   - Uses TF-IDF vectorization of book features
   - K-Nearest Neighbors for finding similar books
   - Matches based on title, author, and genre

2. **Collaborative Filtering**
   - User-item matrix (pivot table) analysis
   - Cosine similarity between books
   - Fallback when content-based fails

### Search Strategy

1. **Exact Match**: Direct lookup in book indices
2. **Keyword Match**: Partial matching with all keywords present
3. **Fuzzy Match**: Close matches using difflib (70% similarity threshold)
4. **Collaborative Fallback**: Uses similarity scores from user ratings

## API Reference

### `hybrid_recommend_clean(book_query, n=5)`

Main recommendation function used by both interfaces.

**Parameters:**
- `book_query` (str): Book title, author, or genre to search for
- `n` (int): Number of recommendations to return (default: 5)

**Returns:**
- List of dictionaries containing book information:
  ```python
  [
      {
          "Title": "Book Name",
          "Author": "Author Name", 
          "Genre": "Genre",
          "Image-URL-M": "cover_image_url"
      }
  ]
  ```

### `chatbot_response(user_input, history)`

Processes user input through Watson Assistant and triggers recommendations.

**Parameters:**
- `user_input` (str): User's message
- `history` (list): Conversation history

**Returns:**
- Updated conversation history and empty input field

## Customization

### Adding New Books
1. Update the source dataset
2. Regenerate the pickle files:
   - `books_cleaned.pkl`
   - `indices.pkl` 
   - `tfidf_matrix.pkl`
   - `similarity_scores.pkl`

### Modifying UI
- Edit CSS in the `gr.Blocks()` section of `book_recommender.py`
- Replace images in the `static/` directory
- Adjust Gradio components for different layouts

### Watson Configuration
- Train custom intents in IBM Watson Assistant
- Modify trigger keywords in the `trigger_keywords` list
- Customize response formatting in `chatbot_response()`

## Troubleshooting

### Common Issues

**"File not found" errors:**
- Ensure all `.pkl` files are in the correct `models/` directory
- Check file paths are relative to the script location

**Watson API errors:**
- Verify API credentials are correct and active
- Check internet connection for Watson API calls
- Ensure workspace ID matches your Watson Assistant

**No recommendations returned:**
- Try broader search terms
- Check if the query book exists in the dataset
- Verify model files are properly loaded

**Gradio UI issues:**
- Ensure static files exist in the `static/` directory
- Check file permissions for image assets
- Try running with `demo.launch(share=True)` for external access

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request



## Acknowledgments

- IBM Watson Assistant for natural language processing
- Gradio for the web interface framework
- Scikit-learn for machine learning algorithms
- The open-source book dataset used for training
