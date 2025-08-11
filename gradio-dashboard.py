import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_available.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=0)
documents = [d for d in text_splitter.split_documents(raw_documents) if d.page_content.strip()]


emb = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=128)
db_books = Chroma(collection_name="books", embedding_function=emb, persist_directory="chroma_books")

texts = [d.page_content for d in documents]
BATCH = 100
for i in range(0, len(texts), BATCH):
    db_books.add_texts(texts[i:i+BATCH])

try:
    db_books._client.persist()
except AttributeError:
    pass



def retrieve_semantic_recommendations(
        query:str,
        category:str = None,
        tone:str = None,
        initial_top_k:int = 50,
        final_top_k:int = 16,
) ->pd.DataFrame:

    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(doc.page_content.strip('"').split()[0]) for doc, _ in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs = books_recs.sort_values(by="joy", ascending=False, ignore_index=True)
    elif tone == "Surprising":
        books_recs = books_recs.sort_values(by="surprise", ascending=False, ignore_index=True)
    elif tone == "Angry":
        books_recs = books_recs.sort_values(by="anger", ascending=False, ignore_index=True)
    elif tone == "Suspenseful":
        books_recs = books_recs.sort_values(by="fear", ascending=False, ignore_index=True)
    elif tone == "Sad":
        books_recs = books_recs.sort_values(by="sadness", ascending=False, ignore_index=True)

    return books_recs

def recommend_books(
        query:str,
        category:str,
        tone:str
):

    recommendations = retrieve_semantic_recommendations(query,category,tone)
    result = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        result.append([row["large_thumbnail"], caption])
    return result

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy","Surprising","Angry","Suspenseful","Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommendations")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a Category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books",columns=8,rows=2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query,category_dropdown,tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
