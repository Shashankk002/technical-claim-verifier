import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()

with open('./data/abstracts.json', 'r') as file:
    paper_data = json.load(file)

abstracts = [paper["abstract"] for paper in paper_data]
tfidf_matrix = vectorizer.fit_transform(abstracts)

claim = "Scaling always improves reasoning performance."
claim_vector = vectorizer.transform([claim])


# similarities = [cosine_similarity(claim_vector, abstract_vector) for abstract_vector in tfidf_matrix]
similarities = cosine_similarity(claim_vector, tfidf_matrix)

#ranked_indices = sorted(range(len(similarities[0])), key=lambda i: similarities[0][i], reverse=True)
ranked_indices = np.argsort(similarities[0])[::-1]  #-1 to reverse it

for i in range(3):
    print(f"Rank = {i + 1}")
    idx = ranked_indices[i]
    print(f"Title: {paper_data[idx]["title"]}")
    print(f"Similarity score: {similarities[0][idx]:.4f}")
    if(i!=2): print()



