import os
import pickle
import numpy as np
import pandas as pd
import re
from difflib import get_close_matches
from ibm_watson import AssistantV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import gradio as gr

# ==== Resolve absolute paths ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ==== Load Pickle Files ====
def load_pickle(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

books_cleaned = load_pickle("books_cleaned.pkl")
indices = load_pickle("indices.pkl")
nn_model = load_pickle("nn_model.pkl")
tfidf_matrix = load_pickle("tfidf_matrix.pkl")
similarity_scores = load_pickle("similarity_scores.pkl")
pt = load_pickle("pt.pkl")

# ==== Utility Functions ====
def normalize_title(title):
    return re.sub(r"\s*\(.?\)\s", "", title).strip().lower()

def extract_keywords_for_recommendation(user_input):
    text = user_input.lower()
    for keyword in ["like", "recommend me", "suggest", "books on", "books like"]:
        if keyword in text:
            return user_input.split(keyword)[-1].strip()
    return user_input.strip()

# ==== Hybrid Recommender ====
def hybrid_recommend_clean(book_query, n=5):
    book_query = book_query.strip().lower()
    content_found = False

    if book_query in indices:
        idx = indices[book_query]
        content_found = True
    else:
        keywords = book_query.split()
        matches = books_cleaned['Title'].str.lower()
        partial = matches[matches.apply(lambda t: all(k in t for k in keywords))]
        if not partial.empty:
            idx = indices[partial.index[0]]
            content_found = True
        else:
            close = get_close_matches(book_query, indices.index, n=1, cutoff=0.7)
            if close:
                idx = indices[close[0]]
                content_found = True

    if content_found:
        fetch_n = n * 5
        _, neighbors = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=fetch_n + 1)
        rec = books_cleaned.iloc[neighbors[0][1:]][['Title', 'Author', 'Genre', 'Image-URL-M']].copy()
        rec['norm'] = rec['Title'].apply(normalize_title)
        rec = rec.drop_duplicates(subset='norm').drop(columns=['norm'])
        return rec.head(n).to_dict(orient="records")

    norm_index = pt.index.str.strip().str.lower()
    matches = [title for title in norm_index if book_query in title]
    if matches:
        matched = matches[0]
        idx = np.where(norm_index == matched)[0][0]
        similar = sorted(list(enumerate(similarity_scores[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
        titles = [pt.index[i[0]] for i in similar]
        return [{"Title": t} for t in titles]

    return [{"message": "No similar books found."}]

# ==== IBM Watson Assistant Setup ====
API_KEY = "E9SsLc77focYBnqboMZZ5kAlVDZ4pCN8UPbOWy9hhcAm"
ASSISTANT_ID = "42d32f55-a791-46fc-8ef9-4f3796cbcadd"
URL = "https://api.us-south.assistant.watson.cloud.ibm.com"

authenticator = IAMAuthenticator(API_KEY)
assistant = AssistantV1(version="2021-06-14", authenticator=authenticator)
assistant.set_service_url(URL)
workspace_id = ASSISTANT_ID

# ==== Chatbot Logic ====
def chatbot_response(user_input, history):
    try:
        response = assistant.message(
            workspace_id=workspace_id,
            input={"text": user_input}
        ).get_result()

        watson_reply = response["output"].get("text", ["🤖 I'm not sure how to respond to that."])[0]
        history.append((user_input, watson_reply))

        trigger_keywords = ["recommend", "suggest", "fantasy", "thriller", "romance", "novel", "science fiction", "mystery", "horror", "book", "read", "harry potter", "game of thrones", "books like", "books on"]
        if any(kw in user_input.lower() for kw in trigger_keywords):
            query = extract_keywords_for_recommendation(user_input)
            recommendations = hybrid_recommend_clean(query)

            rec_text = f"\n📚 Book Recommendations for: *{query}*\n"
            for i, rec in enumerate(recommendations, 1):
                if "message" in rec:
                    rec_text += f"{i}. {rec['message']}\n"
                else:
                    rec_text += f"\n{i}. *{rec['Title']}*\n"
                    rec_text += f"   Author: {rec.get('Author', 'Unknown')}\n"
                    rec_text += f"   Genre: {rec.get('Genre', 'Unknown')}\n"
                    if 'Image-URL-M' in rec and pd.notna(rec['Image-URL-M']):
                        rec_text += f"   ![]({rec['Image-URL-M']})\n"

            history.append(("📚 Recommendation", rec_text))

    except Exception as e:
        history.append(("error", f"❌ Watson Error: {e}"))

    return history, ""

# ==== Gradio UI ====
background_path = "/file/static/background.jpg"
dobby_path = "/file/static/dobby.png"

with gr.Blocks(css=f"""
body {{
    background-image: url('{background_path}');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

#dobby {{
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 180px;
    margin-bottom: 20px;
}}

#chatbox {{
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 20px;
}}
""") as demo:

    gr.HTML(f"<img id='dobby' src='{dobby_path}' alt='Dobby'>")
    gr.Markdown("## 👋 Welcome to ThinkShelf – Your Book Companion!")

    chatbot = gr.Chatbot(elem_id="chatbox")
    msg = gr.Textbox(label="Ask Dobby for a book recommendation")
    state = gr.State([])

    msg.submit(chatbot_response, [msg, state], [chatbot, msg])

demo.launch()