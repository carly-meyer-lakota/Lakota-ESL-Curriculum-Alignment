import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# Load the CSV file
@st.cache_data
def load_data():
    return pd.read_csv("reach higher curriculum all units - MASTER - reach higher curriculum all units.csv")

df = load_data()

# Define keyword expansion for topic search
def expand_keywords(term):
    keyword_map = {
        'energy': ['energy', 'power', 'electricity', 'force', 'motion', 'fuel', 'heat'],
        'weather': ['weather', 'climate', 'storm', 'rain', 'temperature', 'wind'],
        'community': ['community', 'neighborhood', 'citizen', 'volunteer', 'help'],
        # Add more topic expansions as needed
    }
    return keyword_map.get(term.lower(), [term])

# Define topic search with token matching and fuzzy tiebreaker
def topic_search(term):
    expanded_terms = expand_keywords(term)
    df['combined_words'] = df['Vocabulary Words'].fillna('') + ' ' + df['Related Words'].fillna('')
    
    def compute_overlap_and_fuzzy(text):
        tokens = text.lower().split()
        overlap = sum(1 for t in expanded_terms if t in tokens)
        fuzzy_score = max(fuzz.token_set_ratio(t.lower(), text.lower()) for t in expanded_terms)
        return pd.Series([overlap, fuzzy_score])
    
    df[['overlap_count', 'fuzzy_score']] = df['combined_words'].apply(compute_overlap_and_fuzzy)
    top_matches = df.sort_values(by=['overlap_count', 'fuzzy_score'], ascending=False).head(5)
    return top_matches[['RH Level', 'Unit', 'Part ', 'Unit Name', 'Vocabulary Words']]

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
st.title("📚 Reach Higher Curriculum Search Tool")

search_type = st.selectbox("What would you like to search by?", ["Topic", "Skill", "Genre"])
search_term = st.text_input("Enter your search term:")

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
