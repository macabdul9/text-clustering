# Install the required libraries
# pip install nomic datasets transformers torch pandas numpy

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from nomic import atlas

# Step 1: Load the dataset from Hugging Face
dataset = load_dataset('SynthD/topics', split='train')

# Extract the 'topic' column data from the dataset
texts = dataset['topic']  # Assuming 'topic' is a text field

# Step 2: Load the pre-trained Hugging Face model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # You can choose any other transformer model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Step 3: Generate embeddings for the text data
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Average pooling over the token embeddings to get a fixed-size embedding for each text
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# We might need to process texts in batches to avoid memory issues for larger datasets
batch_size = 64
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = get_embeddings(batch_texts)
    all_embeddings.append(batch_embeddings)

# Concatenate all batches into a single tensor
embeddings = torch.cat(all_embeddings)

# Convert embeddings to numpy array for Nomic Atlas
embeddings_numpy = embeddings.cpu().numpy()

# Step 4: Create a DataFrame for your text and embeddings
df = pd.DataFrame({'text': texts, 'embedding': list(embeddings_numpy)})

# Step 5: Use Nomic Atlas to visualize and cluster the embeddings
project = atlas.map_embeddings(
    embeddings=embeddings_numpy,  # Pass the embeddings as a numpy array
    data=df['text'],  # The original text data associated with each embedding
    name="Topic Clustering from SynthD Dataset"
)

# Step 6: Show the interactive map in your browser
project.show()
