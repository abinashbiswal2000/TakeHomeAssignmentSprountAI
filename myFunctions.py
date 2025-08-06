import io
import docx2txt
from PyPDF2 import PdfReader

def extract_text_from_file(uploaded_file):
    """
    Extracts text from a Streamlit UploadedFile object depending on its type.
    Supports .txt, .pdf, .docx
    """
    file_type = uploaded_file.type

    # Handle TXT
    if file_type == "text/plain":
        return uploaded_file.read().decode("utf-8")

    # Handle DOCX
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return docx2txt.process(uploaded_file)

    # Handle PDF
    elif file_type == "application/pdf":
        text = ""
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    else:
        return ""  # Unsupported format fallback


def build_corpus(jd_text, uploaded_files):
    """
    Constructs the corpus: [job_description, resume_1, resume_2, ...]
    """
    corpus = [jd_text.strip()]

    for file in uploaded_files:
        content = extract_text_from_file(file)
        corpus.append(content.strip())

    return corpus


# -------------------------------------------------------


import nltk

# Check and download 'punkt'
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Check and download 'stopwords'
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Make sure required NLTK resources are available
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocessCorpus(corpus):
    """
    Takes a list of strings (documents), and returns a list of cleaned, preprocessed strings.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    processed_corpus = []

    for doc in corpus:
        # 1. Lowercase
        doc = doc.lower()

        # 2. Remove special characters, numbers, symbols (keep only words)
        doc = re.sub(r'[^a-z\s]', ' ', doc)

        # 3. Tokenize
        tokens = word_tokenize(doc)

        # 4. Remove stopwords and short words, apply stemming
        cleaned_tokens = [
            stemmer.stem(word) for word in tokens
            if word not in stop_words and len(word) > 2
        ]

        # 5. Join tokens back into string
        cleaned_text = ' '.join(cleaned_tokens)
        processed_corpus.append(cleaned_text)

    return processed_corpus






# ----------------------------------------------------------



from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(corpus):
    """
    Takes a list of preprocessed strings and returns the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)  # Returns a sparse matrix
    return tfidf_matrix







# ------------------------------------------------

from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculates cosine similarity between the JD vector (first row) and all resume vectors (rest).
    Returns a 1D list of similarity scores.
    """
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    cosine_scores = cosine_similarity(jd_vector, resume_vectors)[0]  # Shape: (N,)
    return cosine_scores
