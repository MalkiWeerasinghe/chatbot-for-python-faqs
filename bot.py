import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Load the FAQ dataset
with open("faq_dataset.json", "r") as file:
    faq_data = json.load(file)["faqs"]

# Preprocess FAQ questions
faq_questions = [preprocess_text(item["question"]) for item in faq_data]
faq_answers = [item["answer"] for item in faq_data]

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(faq_questions)
faq_vectors = vectorizer.transform(faq_questions)

# Chatbot function
def chatbot_response(user_query):
    # Preprocess the user query
    query = preprocess_text(user_query)
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_vector, faq_vectors)
    best_match_index = similarities.argmax()
    best_match_score = similarities[0, best_match_index]
    
    # Set a threshold for matching
    if best_match_score > 0.2:  # Adjust threshold as needed
        return faq_answers[best_match_index]
    else:
        return "Sorry, I don't understand your question. Can you rephrase?"

# Test the chatbot
print("Python FAQ Chatbot. Type 'exit' or 'quit' to end.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")

