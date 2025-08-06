# myFunctionsEmbeddings.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def build_corpus(jd_text, uploaded_files):
    """
    Constructs the corpus: [job_description, resume_1, resume_2, ...]
    """
    from myFunctions import extract_text_from_file  # reuse the function from your other file

    corpus = [jd_text.strip()]
    for file in uploaded_files:
        content = extract_text_from_file(file)
        corpus.append(content.strip())

    return corpus


def make_embeddings(corpus):
    """
    Generates sentence embeddings using a pre-trained transformer model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_tensor=False)
    return embeddings


def calculate_cosine_embeddings(embeddings):
    """
    Calculates cosine similarity between JD embedding and resume embeddings.
    Returns a 1D list of similarity scores.
    """
    jd_embedding = embeddings[0].reshape(1, -1)
    resume_embeddings = embeddings[1:]
    cosine_scores = cosine_similarity(jd_embedding, resume_embeddings)[0]
    return cosine_scores
