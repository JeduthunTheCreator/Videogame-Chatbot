#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design 
"""
import wikipedia
import requests
import aiml
import pandas as pd
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import numpy as np
from tensorflow.keras.models import load_model

# Load the Q/A dataset
qa_data = pd.read_csv("C:\\Users\\idemu\\OneDrive\\Desktop\\Final Year Projects\\AI\\AI_sample pairs.csv")

# Ensure that 'Question' and 'Answer' columns exist
if 'Questions' not in qa_data.columns or 'Answers' not in qa_data.columns:
    raise ValueError("The data file does not contain the required 'Question' and 'Answer' columns.")


# Function to preprocess text
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    return text


# Applying preprocessing to Questions and Answers columns
qa_data['Questions'] = qa_data['Questions'].apply(preprocess)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(qa_data['Questions'])


def find_most_similar_question(userInput):
    # Preprocess the user's question
    processed_question = preprocess(userInput)

    # Convert the user's question to a vector
    userInput_vector = vectorizer.transform([processed_question])

    # Compute cosine similarity between user's question and all questions in the knowledge base
    similarities = cosine_similarity(userInput_vector, X)

    # Find the index of the most similar question
    most_similar_question_index = similarities.argmax()
    similarity_score = similarities[0, most_similar_question_index]

    return qa_data.iloc[most_similar_question_index]['Answers']


def get_games(genre, game_type, api_key):
    base_url = "https://api.rawg.io/api/games"
    params = {
        'key': "82534973e22d4a329cb498555c38c5c8",
        'genres': genre,
        'ordering': '-added' if game_type == 'popular' else '-rating'
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            game_data = response.json()
            games = [game['name'] for game in game_data['results'][:8]]
            return 'Here are some {} {} games: {}'.format(game_type, genre, ', '.join(games))
        else:
            return f"Failed to fetch game data: Status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching game data: {str(e)}"


# Load the model
model = load_model("C:\\Users\\idemu\\OneDrive\\Desktop\\Final Year Projects\\AI\\Gameplay_model.h5")


def classify_game_image(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(640, 360), color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)
        game_index = np.argmax(predictions[0])
        games = ['Among Us', 'Apex Legends', 'Fortnite', 'Forza Horizon', 'Free Fire', 'Genshin Impact', 'God of War', 'Minecraft', 'Roblox', 'Terraria0']
        return games[game_index]
    except Exception as e:
        return f"An error occurred while classifying the image: {str(e)}"


# Initialize NLTK Inference and read expressions
read_expr = Expression.fromstring


# Load the Knowledgebase from a CSV file
def load_knowledge_base(filename):
    kb = []
    try:
        data = pd.read_csv(filename, header=None, encoding='ISO-8859-1')
        for row in data[0]:
            kb.append(read_expr(row))
        return kb
    except Exception as e:
        print(f"An error occurred while loading the knowledge base: {str(e)}")
        return []


# Load the knowledge base
kb = load_knowledge_base('C:\\Users\\idemu\\OneDrive\\Desktop\\Final Year Projects\\AI\\VGknowledgebase.csv')


# AIML Kernel Initialization
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="C:\\Users\\idemu\\OneDrive\\Desktop\\Final Year Projects\\AI\\mybot-basic.xml")

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")

# Load API key from an environment variable or a configuration file
rawg_api_key = os.getenv('RAWG_API_KEY')

# Main loop
while True:
    # get user input
    try:
        userInput = input("> ")
        if userInput.lower() == "what game is this?":
            imagePath = input("Please enter the path to the game image: ")
            try:
                game = classify_game_image(imagePath, model)
                print(f"This is {game}.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
            continue
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

    aimlInput = userInput
    answer = kern.respond(aimlInput)

    if answer and answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])

    # post-process the answer for commands
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                print(wSummary)
            except Exception as e:
                print("Sorry, I do not know that. Be more specific!")
        elif cmd == 2:
            genre = params[1].lower()
            game_type = 'popular'
            print(get_games(genre, game_type, rawg_api_key))
        elif cmd == 3:
            genre = params[1].lower()
            game_type = 'indie'
            print(get_games(genre, game_type, rawg_api_key))
        elif cmd == 31:
            object, subject = params[1].split(' is ')
            object = object.replace(' ', '_')  # Replace spaces with underscores
            predicate_name = ''.join(word.capitalize() for word in subject.split())
            expr_str = f"{predicate_name}({object})"
            neg_expr_str = f"~{expr_str}"

            # Creating the expressions
            expr = read_expr(expr_str)
            neg_expr = read_expr(neg_expr_str)

            # Check if there is a contradiction in the knowledge base
            if neg_expr in kb:
                print(f"Contradiction detected: {object.replace('_', ' ')} cannot be {subject}.")
            else:
                kb.append(expr)  # Add the fact
                print(f"OK, I'll remember that {object.replace('_', ' ')} is a {subject}.")
        elif cmd == 32:
            object, subject = params[1].split(' is ')
            object = object.replace(' ', '_')
            predicate_name = ''.join(word.capitalize() for word in subject.split())
            expr_str = f"{predicate_name}({object})"
            expr = read_expr(expr_str)

            # Checking the expression against the knowledge base
            if expr in kb:
                print('Correct.')
            else:
                neg_expr_str = f"~{expr_str}"
                neg_expr = read_expr(neg_expr_str)
                if neg_expr in kb:
                    print('Incorrect.')
                else:
                    print('Sorry, I don\'t know.')
        elif cmd == 99:
            answer = find_most_similar_question(userInput)
            print(answer)
    else:
        if not answer:
            answer = find_most_similar_question(userInput)
        print(answer)
