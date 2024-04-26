from flask import Flask, render_template, request
import numpy as np
import re
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection(name="embedding_data")
ids = collection.get()
embeddings = collection.get(ids=ids['ids'], include=["embeddings", "metadatas"])
data = embeddings['metadatas']
names = [item['name'] for item in data]

bert_vector_list = embeddings['embeddings']
bert_vector = np.array(bert_vector_list)

model = SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route('/display', methods=["GET", "POST"])
def display():
    text = request.form.get("text")
    input_embedding = model.encode(text)
    cos_similarities = cosine_similarity([input_embedding], bert_vector)
    top_indices = cos_similarities.argsort()[0][-10:][::-1]

    unique_filenames = []
    for i in top_indices:
        if i < len(names):  # Check if index is within the range of filenames
            filename = names[i]
            if filename not in unique_filenames:
                unique_filenames.append(filename)
            if len(unique_filenames) == 10:
                break

    return render_template("display.html", filenames=unique_filenames)


if __name__ == '__main__':
    app.run(debug=True)
