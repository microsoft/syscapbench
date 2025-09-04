"""Evaluator class for evaluating the performance of the model."""

import re

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sdk.llm_evaluator import LLMJudger


class Evaluator:
    """Evaluator class for evaluating the performance of the model."""

    def __init__(self, _model_name='gpt-4o') -> None:
        """Initialize the Evaluator class."""
        self.model_name = _model_name

    def syntax_correctness(self, query):
        """Check the syntax correctness of the query."""
        pattern = r'cluster\(.*?\)\.database\(.*?\)'
        return 1 if re.search(pattern, query) else 0

    def jaccard_similarity_ngrams(self, query1, query2, n=2):
        """Calculate the Jaccard similarity between two queries using n-grams."""
        query_ngrams1 = {query1[i : i + n] for i in range(len(query1) - n + 1)}
        query_ngrams2 = {query2[i : i + n] for i in range(len(query2) - n + 1)}
        intersection = query_ngrams1.intersection(query_ngrams2)
        union = query_ngrams1.union(query_ngrams2)
        similarity = len(intersection) / len(union)
        return similarity

    def cosine_similarity_strings(self, query1, query2):
        """Calculate the cosine similarity between two strings."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query1, query2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def embeddings_similarity(self, query1, query2):
        """Calculate the cosine similarity between two queries using embeddings."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = [query1, query2]
        embeddings = model.encode(sentences)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def exact_match(self, query1, query2):
        """Check if the two queries are exactly the same."""
        query1 = query1.strip()
        query2 = query2.strip()
        return 1 if query1 == query2 else 0

    def offline_evaluate(self, question, answer, groundtruth):
        """Evaluate the performance of the model."""
        syntax_acc = self.syntax_correctness(answer)
        em = self.exact_match(answer, groundtruth['response'])
        sim = self.jaccard_similarity_ngrams(answer, groundtruth['response'])
        cos_sim = self.cosine_similarity_strings(answer, groundtruth['response'])
        embeddings_similarity = self.embeddings_similarity(answer, groundtruth['response'])
        llmjudger = LLMJudger(self.model_name)
        llmjudger_answer, llmjudger_rating = llmjudger.run(
            question=question, answer=answer, ref_answer=groundtruth['response']
        )
        return (
            syntax_acc,
            em,
            sim,
            cos_sim,
            embeddings_similarity,
            llmjudger_rating,
            llmjudger_answer,
        )
