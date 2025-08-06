import streamlit as st
import pandas as pd
from myFunctions import build_corpus , preprocessCorpus , vectorize , calculate_cosine_similarity


# Set page title
st.set_page_config(page_title="Assignment", layout="centered")

st.title("Candidate Recommendation Engine")


# Job Description input
jd_text = st.text_area("Enter Job Description", height=200, placeholder="Paste the job description here...")

# Resume uploads
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)



# Buttons for processing
st.markdown("### Choose Matching Method")
tfidf_button = st.button("Match using TF-IDF Vectorization", use_container_width=True)
embedding_button = st.button("Match using Transformer based Sentence Embeddings", use_container_width=True)




if tfidf_button:
    if not jd_text.strip():
        st.toast("Please enter a job description.", icon="⚠️")
    elif not uploaded_files:
        st.toast("Please upload at least one resume file.", icon="📄")
    else:
        corpus = build_corpus(jd_text, uploaded_files)
        cleanedCorpus = preprocessCorpus(corpus)
        tfidf_matrix = vectorize(cleanedCorpus)
        scores = calculate_cosine_similarity(tfidf_matrix)        
        

        st.subheader("🏆 Top 5 Resume Matches (TF-IDF)")

        # Sort scores in descending order
        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        # Prepare top 5 data
        top_5 = [
            {"Name": uploaded_files[i].name, "Similarity Score": f"{score:.2f}"}
            for i, score in sorted_indices[:5]
        ]

        # Create DataFrame
        df = pd.DataFrame(top_5)

        # Display with no index
        st.dataframe(df, use_container_width=True, hide_index=True)

        






from myFunctionsEmbeddings import build_corpus as build_corpus_embed
from myFunctionsEmbeddings import make_embeddings, calculate_cosine_embeddings

if embedding_button:
    if not jd_text.strip():
        st.toast("Please enter a job description.", icon="⚠️")
    elif not uploaded_files:
        st.toast("Please upload at least one resume file.", icon="📄")
    else:
        corpus = build_corpus_embed(jd_text, uploaded_files)
        embeddings = make_embeddings(corpus)
        scores = calculate_cosine_embeddings(embeddings)

        st.subheader("🏆 Top 5 Resume Matches (Embeddings)")

        # Sort scores in descending order
        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        # Prepare top 5 data
        top_5 = [
            {"Name": uploaded_files[i].name, "Similarity Score": f"{score:.2f}"}
            for i, score in sorted_indices[:5]
        ]

        # Create DataFrame
        df = pd.DataFrame(top_5)

        # Display with no index
        st.dataframe(df, use_container_width=True, hide_index=True)
