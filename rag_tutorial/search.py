import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS 

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
index_folder_path = 'faiss_index'

def search_local(index_name, question):
    embeddings = OpenAIEmbeddings()
    # load stored vector database
    db = FAISS.load_local(index_folder_path, embeddings, index_name)

    # search 
    retrieved_docs = db.similarity_search(question)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)

    prompt = '''
        Question: {}\n
        Documents: {}\n
        Answer is: 
    '''.format(question, retrieved_docs)

    print(f'PROMPT: {prompt}\n')
    answer = (llm.invoke(prompt)).strip()
    return answer

def main():
    index_name = 'soft_arch'
    while True:      
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        user_input = input("Enter your question: ")
        output = search_local(index_name, user_input)
        print(f"Answer is: {output}")

if __name__ == "__main__":
    main()