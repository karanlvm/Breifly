import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import time

# Initialize the Gemini API
def initialize_gemini():
    # Configure the API key
    genai.configure(api_key= "..")  # Replace with your actual API key

    # Create the model with specified configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction=(
            "I will give you news articles. Summarize each article so an average human being "
            "can read it in under 60 seconds. In a new line print - "
            "trust score (numerical value), source if available,  sentiment analysis (positive, neutral, or negative), "
            "and political alignment (left, center, or right)."
        ),
    )

    # Start a chat session
    chat_session = model.start_chat(history=[])
    
    return chat_session

# Fetch response from Gemini
def get_gemini_response(chat_session, description, retries=3):
    try:
        response = chat_session.send_message(description)
        return response
    except Exception as e:
        st.write(f"Error: {str(e)}")
        # Retry logic if there is an error
        if retries > 0:
            st.write("Retrying request after 5 seconds...")
            time.sleep(5)
            return get_gemini_response(chat_session, description, retries - 1)
        else:
            st.write("Max retries reached. Skipping this request.")
            return None

# Load the CSV file directly from the code
def load_news_data():
    # Load the CSV file (replace 'news_data.csv' with your actual file path)
    data = pd.read_csv('news_data.csv')
    return data[['title', 'description', 'content']]

# Streamlit Application
def main():
    st.title("News Articles with Gemini Analysis")

    # Load the CSV data with titles and descriptions
    news_data = load_news_data()

    # Initialize Gemini API session
    chat_session = initialize_gemini()

    # Control the number of articles processed
    requests_made = 0
    total_articles = len(news_data)

    # Loop through the articles
    for index, row in news_data.iterrows():
        st.write(f"### News Article {index + 1}")

        # Display the title and description
        st.subheader(row['title'])

        # Pass the content to Gemini for response
        gemini_response = get_gemini_response(chat_session, row['content'])

        if gemini_response:
            # Display Gemini response (summary, trust score, sentiment, and political alignment)
            st.write(f"{gemini_response.text}")

        # Increment the requests counter
        requests_made += 1

        # Introduce a delay after every request to slow down the process
        st.write("Waiting for 10 seconds after this request...")
        time.sleep(10)  # 10-second delay after every request

        # After every 3 requests, introduce a longer delay
        if requests_made % 3 == 0:
            st.write("Waiting for 30 seconds after every 3 requests...")
            time.sleep(30)  # 30-second delay after every 3 requests

        # Stop after 15 articles (limit for one minute)
        if requests_made >= 15:
            st.write("Processed 15 articles. Waiting for 60 seconds before continuing...")
            time.sleep(60)  # 60-second delay before processing the next batch
            requests_made = 0

# Run the Streamlit application
if __name__ == "__main__":
    main()