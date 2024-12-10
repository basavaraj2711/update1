import streamlit as st
import PyPDF2
import openai
from gtts import gTTS
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"

# Load Hugging Face pipeline for summarization and NER
summarizer = pipeline("summarization")
ner_tagger = pipeline("ner")

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

# Function to summarize text using OpenAI API (alternative to Gemini)
def analyze_text_with_openai(text):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Summarize this text. Extract key data points, entities, relationships, and actionable insights:\n\n{text}",
            temperature=0.5,
            max_tokens=1000
        )
        summary = response.choices[0].text.strip()
        return summary
    except Exception as e:
        return f"Error with OpenAI API: {str(e)}"

# Function to extract entities using Hugging Face's NER pipeline
def extract_entities_and_relationships(text):
    entities = ner_tagger(text)
    extracted_entities = [(ent['word'], ent['entity']) for ent in entities]
    relations = []  # Placeholder, for now, relationships can be further refined
    return extracted_entities, relations

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

    st.write("Summarizing text and extracting key data using OpenAI API...")
    analysis = analyze_text_with_openai(pdf_text)
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
