
# 📚 Book Recommender System


## Overview
This project is a **Book Recommender System** built with Python, combining **data exploration, machine learning, natural language processing (NLP)**, and **interactive dashboards**.  
It processes a large dataset of books, cleans and enriches the data, performs **text classification, sentiment analysis, emotion detection**, and finally presents results via an interactive **Gradio dashboard**.

---

## 1. Installation & Setup

### Step 1: Install PyCharm
We use **PyCharm** as our main IDE for development.

### Step 2: Install Required Libraries
```bash
pip install kagglehub pandas matplotlib seaborn python-dotenv langchain-community langchain-openai langchain-chroma transformers gradio notebook ipywidgets torch torchvision torchaudio tqdm
```

**Libraries and their purposes:**
- **Kagglehub** → Download datasets directly from Kaggle.
- **Pandas** → Handle tabular data.
- **Matplotlib** & **Seaborn** → Data visualization.
- **Python-dotenv** → Store API keys securely.
- **Langchain** → Work with LLMs and vector databases.
- **Transformers** → Pre-trained NLP models.
- **Gradio** → Create interactive dashboards.
- **Notebook** & **ipywidgets** → Jupyter interactivity.
- **PyTorch** → Deep learning backend.
- **tqdm** → Progress bars.

---

## 2. Data Download & Exploration

### Download the dataset from Kaggle:
```python
import kagglehub

path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
print("Path to dataset files:", path)
```

### Load the dataset into Pandas:
```python
import pandas as pd
df = pd.read_csv(f"{path}/books.csv")
df.head()
```

### Column Statistics & Descriptive Analysis
- View **column statistics** in Jupyter (`Details` tab).
- Count **unique books** → `6398` unique out of `6810` total.
- Most popular book: **The Lord of the Rings**.
- Most common genre: **Fantasy**.
- Most frequent author: **Agatha Christie**.

---

## 3. Data Cleaning
- Identify **missing values** using `seaborn` heatmap.
- Create **new columns**: page count, book age, missing description flag, etc.
- Remove short descriptions (< 25 words).
- Merge columns (title + subtitle, ISBN + description).
- Drop unnecessary columns.

---

## 4. NLP & Machine Learning Components

### 4.1 Word Embeddings & Transformers
- Demonstrate **Word2Vec skipgram** model.
- Use **transformer architecture** with **self-attention**.
- Process text into embeddings with **OpenAI API**.

### 4.2 Vector Search
- Store embeddings in a **Chroma** database.
- Perform **semantic search** on book descriptions.

### 4.3 Text Classification
- Use **Zero-Shot Classification** with HuggingFace models.
- Classify books into genres (Fiction / Non-fiction).
- Achieved **79% accuracy**.

### 4.4 Sentiment Analysis & Emotion Classification
- **Sentiment Analysis** → Detects polarity (positive/negative/neutral).
- **Emotion Classification** → Detects emotions (joy, anger, fear, sadness).
- Run classification **per sentence** in book descriptions.
- Build **emotional profiles** per book.

---

## 5. Dashboard (Gradio)

### Features:
- Semantic search for books.
- Genre & tone filtering.
- Book recommendations with cover images.
- Interactive UI.

```python
import gradio as gr

# Example of creating Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Book Recommender")
    search_query = gr.Textbox(label="Enter a book description")
    genre = gr.Dropdown(["Fiction", "Non-fiction"], label="Genre")
    recommend_button = gr.Button("Recommend")
demo.launch()
```

---

## 6. Running the Project

1. **Download the dataset** from Kaggle using `kagglehub`.
2. **Clean & preprocess** the data in `data-exploration.ipynb`.
3. **Generate embeddings** in `vector-search.py`.
4. **Classify genres** in `classification.py`.
5. **Analyze sentiment & emotions** in `sentiment-analysis.py`.
6. **Launch dashboard** in `gradio-dashboard.py`.

---

## 7. Results & Insights
- **Dataset size**: 6810 books, 6398 unique.
- **Top genre**: Fantasy.
- **Accuracy** of genre classification: 79%.
- **Sentiment & emotion profiles** for all books.
- Fully interactive book recommender system.

---

## 8. Future Improvements
- Add **multi-language support**.
- Implement **collaborative filtering**.
- Improve accuracy with **fine-tuned models**.
- Deploy dashboard to **HuggingFace Spaces**.

---

## Author
**Maor Massas**

## Author
Project developed for educational and demonstration purposes.
