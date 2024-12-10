import streamlit as st
import PyPDF2
import google.generativeai as genai
from gtts import gTTS
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Set your Gemini API key
genai.configure(api_key="YOUR_API_KEY_HERE")

# Load spaCy model for enhanced entity extraction
nlp = spacy.load("en_core_web_trf")  # Transformer-based model for better accuracy

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""  # Handle NoneType gracefully
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Function to summarize text and extract key data using Gemini API
def analyze_text_with_gemini(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"""
            Summarize this text. Extract key data points, entities, relationships, and actionable insights:
            {text}
        """)
        summary = response.text
        return summary
    except Exception as e:
        return f"Error with Gemini API: {str(e)}"

# Function to extract entities and relationships using spaCy
def extract_entities_and_relationships(text):
    doc = nlp(text)
    entities = []
    relations = []

    # Extract named entities
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    # Extract relationships (simple subject-verb-object)
    for sent in doc.sents:
        for token in sent:
            # Fix relationship extraction to handle more cases
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                relations.append((token.text, token.head.text, token.head.dep_))

    return entities, relations

# Function to perform refined word frequency analysis
def refined_word_frequency_analysis(text):
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    word_freq = Counter(words)
    return word_freq.most_common(10)

# Convert text to speech and save it as an MP3 file
def text_to_speech(text, output_file="output.mp3"):
    try:
        tts = gTTS(text, lang="en")
        tts.save(output_file)
        return output_file
    except Exception as e:
        return f"Error converting text to speech: {str(e)}"

# Display knowledge graph using NetworkX and Matplotlib
def display_knowledge_graph(entities, relations):
    G = nx.Graph()
    
    # Add nodes and edges
    for entity, label in entities:
        G.add_node(entity, label=label)
    for subj, verb, obj in relations:
        G.add_edge(subj, obj, relationship=verb)
    
    # Visualize
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
    st.pyplot(plt)

# Streamlit interface
def process_pdf_to_audio_summary(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    if pdf_text.startswith("Error"):
        st.error(pdf_text)
        return

    st.write("Summarizing text and extracting key data using Gemini API...")
    analysis = analyze_text_with_gemini(pdf_text)
    if analysis.startswith("Error"):
        st.error(analysis)
        return

    st.write("Analysis and Summary:")
    st.write(analysis)

    # Convert summary to audio
    st.write("Generating audio summary...")
    audio_file = text_to_speech(analysis)
    if isinstance(audio_file, str) and audio_file.endswith(".mp3"):
        st.audio(audio_file, format="audio/mp3")

    # Perform refined word frequency analysis
    st.write("Performing refined word frequency analysis...")
    word_freq = refined_word_frequency_analysis(pdf_text)
    words, counts = zip(*word_freq)
    fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Frequency'}, title="Refined Word Frequency Analysis")
    st.plotly_chart(fig)

    # Extract entities and relationships
    st.write("Extracting entities and relationships...")
    entities, relations = extract_entities_and_relationships(analysis)

    st.write("Entities:")
    st.write(entities)
    st.write("Relationships:")
    st.write(relations)

    # Display knowledge graph
    st.write("Visualizing knowledge graph...")
    display_knowledge_graph(entities, relations)

# Streamlit UI elements
st.title("Enhanced PDF Analysis and Knowledge Extraction")
st.write("Upload a PDF file to extract its content, analyze it, generate analytics, and create summaries.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the uploaded PDF file
    process_pdf_to_audio_summary("temp_pdf.pdf")
