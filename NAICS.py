import re
import pandas as pd
import os
import requests
import streamlit as st
from nltk.stem import WordNetLemmatizer

# Define a basic list of stopwords
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "mining", "services", "industry", "manufacturing", "and", "or"
])

lemmatizer = WordNetLemmatizer()

# Custom function to tokenize and process text
def simple_tokenize(text):
    words = re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()
    return words

# Define a function to generate keywords from multiple columns using custom tokenization
def process_keywords(*args):
    keywords = []
    for text in args:
        if pd.isna(text):
            continue
        words = simple_tokenize(text)
        keywords.extend([
            lemmatizer.lemmatize(word)
            for word in words if word not in stop_words and len(word) > 2
        ])
    return ' '.join(sorted(set(keywords)))


# URL for the file on Google Drive or your chosen storage
url = 'https://drive.google.com/uc?export=download&id=1ZX35OuMvhaaq4q83E8pI7lqWLAhnkE0B'

# Check if the file already exists
if not os.path.exists("naics_data.csv"):
    response = requests.get(url)
    if response.status_code == 200:
        with open("naics_data.csv", "wb") as file:
            file.write(response.content)
    else:
        st.error("Failed to download naics_data.csv. Please check the file URL or network connection.")

# Load the CSV file
df = pd.read_csv('naics_data.csv')  # Replace with your file path

# Ensure 'Mineral_Type' column has no NaN values
df['Mineral_Type'] = df['Mineral_Type'].fillna('') if 'Mineral_Type' in df.columns else ''

# Define a mineral-to-NAICS mapping
mineral_to_naics_mapping = {
    "lithium": "212398",  # All Other Nonmetallic Mineral Mining And Quarrying
    "gold": "212220",     # Gold And Silver Ore Mining
    "copper": "212230",   # Copper, Nickel, Lead, And Zinc Mining
    "nickel": "212230",
    "zinc": "212230",
    # Add additional mappings as needed
}

# Keep the original combined NAICS code-description format
df['Combined_NAICS'] = df['NAICS_Sector']

# Extract NAICS code and description from 'NAICS_Sector'
df[['NAICS_Code', 'Description']] = df['NAICS_Sector'].str.extract(r'(\d+)\s(.+)', expand=True)

# Fill missing values for 'Description'
df['Description'] = df['Description'].fillna('')

# Ensure specified columns exist; if not, create them with empty values
for column in [
    'Mineral_Type', 'Investing_Company', 'Modified_ICB_Industry',
    'Modified_ICB_Supersector', 'Modified_ICB_Sector', 'Modified_ICB_Subsector', 'Project_Description'
]:
    df[column] = df[column].fillna('') if column in df.columns else ''

# Remove duplicate NAICS-ICB entries
df = df.drop_duplicates(subset=[
    'NAICS_Code', 'Modified_ICB_Industry', 'Modified_ICB_Supersector',
    'Modified_ICB_Sector', 'Modified_ICB_Subsector'
])

# Define a function to generate keywords from multiple columns
def process_keywords(*args):
    keywords = []
    for text in args:
        if pd.isna(text):
            continue
        words = simple_tokenize(re.sub(r'[^a-zA-Z\s]', '', text.lower()))
        keywords.extend([
            lemmatizer.lemmatize(word)
            for word in words if word not in stop_words and len(word) > 2
        ])
    return ' '.join(sorted(set(keywords)))

# Generate 'Searchable_Keywords' by combining keywords from specified columns
df['Searchable_Keywords'] = df.apply(
    lambda row: process_keywords(
        row['Description'], row['Mineral_Type'], row['Investing_Company'],
        row['Modified_ICB_Industry'], row['Modified_ICB_Supersector'],
        row['Modified_ICB_Sector'], row['Modified_ICB_Subsector'], row['Project_Description']
    ), axis=1
)

# Group by NAICS_Code to remove duplicates, keeping the first description and combining keywords
df_grouped = df.groupby('NAICS_Code').agg({
    'Description': 'first',
    'Modified_ICB_Industry': 'first',
    'Modified_ICB_Supersector': 'first',
    'Modified_ICB_Sector': 'first',
    'Modified_ICB_Subsector': 'first',
    'Searchable_Keywords': ' '.join
}).reset_index()

# Add a new column with all Reference_Code values for each NAICS_Code
df_references = df.groupby('NAICS_Code')['Reference_Code'].apply(lambda x: ', '.join(map(str, x))).reset_index()
df_grouped = df_grouped.merge(df_references, on='NAICS_Code', how='left')
df_grouped.rename(columns={'Reference_Code': 'Relevant_Entries'}, inplace=True)

# Streamlit App
st.title("NAICS Code and ICB Search")
st.write("Note: use general, high level searches (i.e. search for 'lithium' not 'lithium mining')")
keyword = st.text_input("Enter a keyword to search NAICS codes:")

if keyword:
    keyword = keyword.lower()
    search_terms = keyword.split()  # Split the search term into individual words

    # Check if the keyword matches a custom mineral-to-NAICS mapping
    if keyword in mineral_to_naics_mapping:
        # Filter for the specific NAICS code associated with the mineral
        naics_code = mineral_to_naics_mapping[keyword]
        results = df_grouped[df_grouped['NAICS_Code'] == naics_code]
    else:
        # Perform an "AND" search on keywords, ensuring each word appears in Searchable_Keywords
        mask = df_grouped['Searchable_Keywords'].apply(
            lambda x: all(term in x for term in search_terms)
        )
        results = df_grouped[mask]

    # Display results
    if not results.empty:
        results['NAICS_Code_Display'] = results.apply(
            lambda row: f"{row['NAICS_Code']} {row['Description']}",
            axis=1
        )
        # Display columns with NAICS Code, Description, Modified ICB classifications, and Relevant Entries
        st.write(results[['NAICS_Code_Display', 'Modified_ICB_Industry',
                          'Modified_ICB_Supersector', 'Modified_ICB_Sector',
                          'Modified_ICB_Subsector', 'Relevant_Entries']])
    else:
        st.write("No results found.")
