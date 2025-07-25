import os
import pickle
import pandas as pd

# ✅ Import hybrid recommender from test.py
from test import hybrid_recommend_clean

# ==== Setup paths ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pt.pkl")  # models/pt.pkl

print(f"Resolved model path: {MODEL_PATH}")

# ==== Load pt.pkl ====
try:
    with open(MODEL_PATH, "rb") as f:
        pt = pickle.load(f)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: File not found. Check your path:", MODEL_PATH)
    pt = None

# ==== Optional: Preview data ====
if pt is not None:
    try:
        print("Sample values from pt:")
        print(pt.head() if hasattr(pt, 'head') else pt[:5])
    except Exception as e:
        print("❌ Error displaying pt:", e)

    print("\nAvailable Book Titles:\n", pt.index[:10])

# ==== Use Recommender ====
if __name__ == "__main__":
    print("\n📚 Welcome to ThinkShelf Book Recommender!")
    query = input("🔎 Enter a book title, author, or genre: ").strip()

    recommendations = hybrid_recommend_clean(query)

    print("\n📖 Top Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        if "message" in rec:
            print(f"{i}. {rec['message']}")
        else:
            print(f"{i}. Title: {rec.get('Title')}")
            print(f"   Author: {rec.get('Author', 'Unknown')}")
            print(f"   Genre: {rec.get('Genre', 'Unknown')}\n")
            
