import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import wordnet
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Automatically download WordNet data if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Load the CSV file
@st.cache_data
def load_data():
    return pd.read_csv("rhallunits.csv")

df = load_data()

# Load sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Expand keywords using WordNet
def expand_keywords(term):
    synonyms = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms) + [term]

# Topic search with token match, fuzzy match, synonym expansion, and semantic similarity
def topic_search(terms):
    all_terms = []
    for term in terms.split(','):
        all_terms.extend(expand_keywords(term.strip()))
    all_terms = list(set(all_terms))

    df['combined_words'] = df['Vocabulary Words'].fillna('') + ' ' + df['Related Words'].fillna('')

    def compute_scores(text):
        tokens = text.lower().split()
        overlap = sum(1 for t in all_terms if t.lower() in tokens)
        fuzzy_score = max(fuzz.token_set_ratio(t.lower(), text.lower()) for t in all_terms)
        embedding_score = util.cos_sim(model.encode(' '.join(all_terms)), model.encode(text)).item()
        return pd.Series([overlap, fuzzy_score, embedding_score])

    df[['overlap_count', 'fuzzy_score', 'embedding_score']] = df['combined_words'].apply(compute_scores)
    df['total_score'] = df['overlap_count'] * 2 + df['fuzzy_score'] + df['embedding_score'] * 100
    top_matches = df.sort_values(by='total_score', ascending=False).head(5)
    return top_matches[['RH Level', 'Unit', 'Part ', 'Unit Name', 'Vocabulary Words', 'total_score']]

# Skill search using fuzzy matching
def skill_search(term):
    skill_columns = ['Language Skill', 'Thinking Map Skill', 'Reading Skill', 'Grammar Skill', 'Project', 'Phonics Skill']
    matches = []

    for col in skill_columns:
        for idx, val in df[col].dropna().items():
            score = fuzz.token_set_ratio(term.lower(), str(val).lower())
            if score > 60:
                matches.append({
                    'RH Level': df.at[idx, 'RH Level'],
                    'Unit': df.at[idx, 'Unit'],
                    'Part ': df.at[idx, 'Part '],
                    'Unit Name': df.at[idx, 'Unit Name'],
                    'Matched Skill Column': col,
                    'Matched Skill Value': val,
                    'Score': score
                })

    top_matches = pd.DataFrame(matches).sort_values(by='Score', ascending=False).head(5)
    return top_matches[['RH Level', 'Unit', 'Part ', 'Unit Name', 'Matched Skill Column', 'Matched Skill Value']]

# Genre search using partial fuzzy match
def genre_search(term):
    df['Genres'] = df['Genres'].fillna('')
    df['relevance'] = df['Genres'].apply(lambda x: fuzz.partial_ratio(term.lower(), x.lower()))
    top_matches = df[df['relevance'] > 60].sort_values(by='relevance', ascending=False).head(5)
    return top_matches[['RH Level', 'Unit', 'Part ', 'Unit Name', 'Genres']]

# Streamlit UI
st.title("📚 ESL Curriculum Search Tool")

search_type = st.selectbox("What would you like to search by?", ["Topic", "Skill", "Genre"])
search_term = st.text_input("Enter your search term(s):")

if st.button("Search") and search_term:
    if search_type == "Topic":
        results = topic_search(search_term)
    elif search_type == "Skill":
        results = skill_search(search_term)
    elif search_type == "Genre":
        results = genre_search(search_term)

    if not results.empty:
        st.write("### Search Results")
        st.dataframe(results)
    else:
        st.warning("No matches found. Try a different search term.")
