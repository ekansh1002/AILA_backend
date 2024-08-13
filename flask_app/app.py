from flask import Flask, request, jsonify
import os
import string
import torch
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from transformers import BertModel, BertTokenizer, BartForConditionalGeneration, BartTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_text

app = Flask(__name__)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load the BART model and tokenizer for summarization
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Directory containing your legal case documents
directory = r"C:\Users\91938\Desktop\dataset\Object_casedocs"

# List to store preprocessed texts and document embeddings
texts = []
document_embeddings = []

def process_documents():
    global texts, document_embeddings
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Assuming documents are in .txt format
            file_path = os.path.join(directory, filename)
            
            # Read the document content
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Preprocess the document text
            preprocessed_text = preprocess_text(text)
            texts.append(preprocessed_text)
            
            # Tokenize and compute embedding for the document
            encoded_text = tokenizer(preprocessed_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoded_text)
                document_embedding = outputs.pooler_output.cpu().detach().numpy()
            
            # Store the document embedding
            document_embeddings.append(document_embedding)

    # Convert document_embeddings list to a numpy array for easier processing
    document_embeddings = np.vstack(document_embeddings)

# Process the documents when the application starts
process_documents()

@app.route('/query', methods=['POST'])
def handle_query():
    global tokenizer, model, bart_tokenizer, bart_model, texts, document_embeddings

    query_text = request.json.get('query')
    if not query_text:
        return jsonify({'error': 'Query text is required.'}), 400

    # Preprocess and tokenize the query
    preprocessed_query = preprocess_text(query_text)
    encoded_query = tokenizer(preprocessed_query, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

    # Process the query with the fine-tuned BERT model to obtain query embedding
    with torch.no_grad():
        outputs = model(**encoded_query)
        query_embedding = outputs.pooler_output.cpu().detach().numpy()  # Get the query embedding

    # Compute cosine similarity between query and document embeddings
    similarity_scores = cosine_similarity(query_embedding, document_embeddings)

    # Rank and retrieve top similar/relevant documents
    top_k = 5  # Number of top documents to retrieve
    top_documents_indices = similarity_scores.argsort()[0][-top_k:][::-1]

    # Generate summaries of top similar documents
    similar_documents = []
    for idx in top_documents_indices:
        document = texts[idx]

        # Generate summary using BART
        inputs = bart_tokenizer([document], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Add similarity score and summary to the list
        similarity_score = similarity_scores[0][idx]
        similar_documents.append({'similarity_score': similarity_score, 'summary': summary})

    return jsonify({'similar_documents': similar_documents})

if __name__ == '__main__':
    app.run(debug=True)
