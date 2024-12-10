### README.md

```markdown
# Enhanced PDF Analysis and Knowledge Extraction

This project provides a comprehensive solution for extracting content from PDF files, summarizing it using AI, performing analytics, and building a knowledge graph. It integrates cutting-edge tools like Google Gemini API, spaCy, Neo4j, and more, to enable automated knowledge graph construction and visualization.

## Features
- **PDF Content Extraction**: Extracts text from uploaded PDF files.
- **AI-Powered Text Analysis**: Summarizes and extracts key insights using the Google Gemini API.
- **Entity and Relationship Extraction**: Uses spaCy to identify entities and relationships within the text.
- **Knowledge Graph Creation**: Uploads extracted data to a Neo4j database and visualizes it as a knowledge graph.
- **Refined Word Frequency Analysis**: Provides a bar chart of the most frequent words in the text after removing stopwords.
- **Text-to-Speech Conversion**: Converts the AI-generated summary into an MP3 audio file for easy consumption.
- **Interactive Visualization**: Displays the knowledge graph using NetworkX and Matplotlib.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.9 or higher
- Neo4j database installed and running
- Required Python libraries (listed in `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/enhanced-pdf-analysis.git
   cd enhanced-pdf-analysis
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Neo4j database and ensure it is accessible at `bolt://localhost:7687`.

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser at the URL provided by Streamlit to interact with the application.

## Configuration
- Set your **Gemini API key** in the code:
  ```python
  genai.configure(api_key="YOUR_API_KEY")
  ```

- Update Neo4j credentials in the code:
  ```python
  NEO4J_URI = "bolt://localhost:7687"
  NEO4J_USERNAME = "neo4j"
  NEO4J_PASSWORD = "password"
  ```

## Usage
1. Upload a PDF file using the Streamlit interface.
2. The app will:
   - Extract text from the PDF.
   - Analyze and summarize the content.
   - Extract entities and relationships.
   - Perform word frequency analysis.
   - Convert the summary to speech.
   - Upload entities and relationships to Neo4j.
   - Visualize the knowledge graph.

## Dependencies
This application uses the following Python libraries:
- Streamlit
- PyPDF2
- google-generativeai
- gTTS
- spaCy
- NetworkX
- Matplotlib
- Plotly
- Neo4j Driver
- TextBlob
- NLTK

## License
This project is open-source and available under the [MIT License](LICENSE).
```

---

### requirements.txt

```plaintext
streamlit==1.27.0
PyPDF2==3.0.1
google-generativeai==0.2.0
gTTS==2.3.2
spacy==3.7.3
networkx==3.0
matplotlib==3.8.0
plotly==5.17.0
neo4j==5.10.0
textblob==0.17.1
nltk==3.9.0
```

With these files, users can easily set up the environment, understand the purpose of the project, and run the application.
