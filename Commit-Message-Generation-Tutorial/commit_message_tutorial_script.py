# -*- coding: utf-8 -*-
"""commit-message-tutorial-script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XWwGORNmr9bW7J9aZx02M3zPQZFW-mtM
"""

# Importing the OpenAI library
from openai import OpenAI
import os
from dotenv import load_dotenv
import csv

load_dotenv()

# Initialize the OpenAI client with your API key
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

#
# THE PROMPT
#  
PROMPT = "Generate a commit message for this diff:\n"

# Read the input file
with open('Commit_Messages.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    n = 2
    for row in reader:
        prompt = PROMPT + row["Code Change"] 

        # Send the user input to OpenAI and get a response
        response = client.chat.completions.create(
            # Specifies the model to be used for the response
            model="gpt-4-1106-preview",

            # The list of messages to be processed by the model
            messages=[
            {
                "role": "user",  # Defines the role of the message sender (user or assistant)
                "content": prompt  # The actual message content
            }
            ],

            # Controls the randomness in the response generation
            # Higher values mean more creative responses
            temperature=1,

            # The maximum number of tokens to generate in the response
            max_tokens=256,

            # Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered
            top_p=1,

            # Adjusts the likelihood of repeating the same line verbatim
            frequency_penalty=0,

            # Penalizes new tokens based on their existing frequency in the text so far
            presence_penalty=0
        )

        # Process and display the response from OpenAI
        if response.choices:
            assistant_message = response.choices[0].message.content
            print(f'Row {n}:\n {assistant_message}\n')
        else:
            # Handle cases where no response is received
            print("No response from the assistant.")
        n += 1

# Print a message when the program is terminated
print("Program exited.")