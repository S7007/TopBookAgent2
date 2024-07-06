# TopBookAgent2
To complete the stage 2 assignment, we will extend the work from stage 1 and incorporate web scraping, comparison, storage in a vector database, and quantization using LoRA. Here's a step-by-step approach to achieving this:

### Step 1: Setup Environment and Tools
We'll use the following tools:
- **Beautiful Soup**: For web scraping.
- **FastAPI**: For creating the REST API.
- **Hugging Face Transformers**: For LLM.
- **FAISS or Pinecone**: For vector database.
- **LoRA**: For quantization of the LLM.
- **Python**: As the primary programming language.

### Step 2: Web Scraping and Comparison
We'll scrape the web content to get book recommendations and compare the results with those from the LLM.

#### Web Scraping Function
```python
import requests
from bs4 import BeautifulSoup

def scrape_top_books(genre):
    url = f"https://www.goodreads.com/shelf/show/{genre}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    books = [book.get_text() for book in soup.find_all("a", class_="bookTitle")]
    return books[:100]
```

#### Comparison Function
```python
def compare_books(llm_books, scraped_books):
    common_books = set(llm_books).intersection(set(scraped_books))
    return list(common_books)
```

### Step 3: Storing Results in Vector Database
We'll use FAISS for this purpose.

#### Setting Up FAISS
```python
import faiss
import numpy as np

# Initialize FAISS index
index = faiss.IndexFlatL2(768)  # 768 is the dimensionality of BERT embeddings
```

#### Function to Store and Retrieve Data
```python
from transformers import AutoTokenizer, AutoModel

# Load tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def store_books(books):
    for book in books:
        embedding = embed_text(book)
        index.add(embedding)

def search_book(book):
    embedding = embed_text(book)
    D, I = index.search(embedding, 1)
    return I[0][0]  # return index of the closest match
```

### Step 4: Extending the REST API
We'll update the FastAPI application to incorporate the new functionalities.

#### Updated FastAPI Application
```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI()
agent = BookAgent()

class GenreRequest(BaseModel):
    genre: str

class BookSelectionRequest(BaseModel):
    book_name: str

@app.post("/top-100-books")
def get_top_100_books(request: GenreRequest):
    llm_books = agent.find_top_100_books(request.genre)
    scraped_books = scrape_top_books(request.genre)
    common_books = compare_books(llm_books, scraped_books)
    store_books(common_books)
    return {"top_100_books": common_books}

@app.get("/top-10-books")
def get_top_10_books():
    books = agent.find_top_10_books()
    return {"top_10_books": books}

@app.post("/select-book")
def select_book(request: BookSelectionRequest):
    selected_book = agent.select_book(request.book_name)
    if not selected_book:
        raise HTTPException(status_code=404, detail="Book not found in top 10")
    index = search_book(request.book_name)
    return {"selected_book": selected_book, "book_index": index}

@app.get("/conclude")
def conclude():
    message = agent.conclude()
    return {"message": message}
```

### Step 5: LoRA Based Quantization
We'll use Low-Rank Adaptation (LoRA) to quantize the model.

#### Quantization Steps
1. **Install LoRA**: Install the necessary package.
2. **Apply LoRA**: Quantize the model and benchmark the performance.

#### Installation and Quantization
```python
!pip install loralib

import loralib as lora
import torch

class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedModel, self).__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return lora(self.model, rank=4)(input_ids, attention_mask, token_type_ids)

# Apply LoRA to the model
quantized_model = QuantizedModel(model)
```

### Step 6: Testing and Documentation

#### Streamlit Application
Update the Streamlit app to reflect the new functionalities.

```python
import streamlit as st
import requests

st.title("Book Selection Agent")

base_url = "http://127.0.0.1:8000"

genre = st.text_input("Enter a genre")
if st.button("Get Top 100 Books"):
    response = requests.post(f"{base_url}/top-100-books", json={"genre": genre})
    st.write(response.json())

if st.button("Get Top 10 Books"):
    response = requests.get(f"{base_url}/top-10-books")
    st.write(response.json())

book_name = st.text_input("Enter the name of the book you want to select")
if st.button("Select Book"):
    response = requests.post(f"{base_url}/select-book", json={"book_name": book_name})
    st.write(response.json())

if st.button("Conclude"):
    response = requests.get(f"{base_url}/conclude")
    st.write(response.json())
```

### Step 7: Finalizing the GitHub Repository
1. **Add all the code**: Include all the scripts for the agent, web scraping, FastAPI, and Streamlit application.
2. **Include the design document**: Document the design choices and the steps for quantization.
3. **Upload the demo video**: Show the interaction with the application.

Here's a checklist to ensure everything is included in the GitHub repository:
- [ ] All source code
- [ ] Design document
- [ ] Demo video

Once completed, share the GitHub repository link.

### Conclusion
This comprehensive guide covers all the necessary steps to complete the stage 2 assignment. Ensure all the functionalities are thoroughly tested and the documentation is clear. Good luck!
