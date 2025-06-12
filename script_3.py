from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load documents
documents = [
    "Python is a programming language used for web development and data analysis.",
    "Machine learning involves algorithms that improve automatically through experience.",
    "PDF files can contain text, images, and other content.",
    "TF-IDF is a method for converting text into numeric features.",
]

# Step 2: Vectorize the documents
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# Step 3: Define a user question
question = "How does machine learning work?"

# Step 4: Transform the question into the same vector space
question_vector = vectorizer.transform([question])

# Step 5: Compute cosine similarity
similarities = cosine_similarity(question_vector, doc_vectors).flatten()

# Step 6: Get the most relevant document
best_match_idx = similarities.argmax()
best_match_score = similarities[best_match_idx]

print("Best answer:")
print(documents[best_match_idx])
print(f"\nSimilarity score: {best_match_score:.2f}")
