# FSE 23 LLMs for Software Engineering


### TODO

- required python version `3.10`

- install required libraries
    ```bash
    pip install -r requirements.txt
    ```

- create `.env` file, put OpenAI API key 
    ```
    OPENAI_API_KEY=''
    ```

### create_index.py
Embeddings create a vector representation of a piece of text. This is useful because it means we can think about text in the vector space, and do things like semantic search where we look for pieces of text that are most similar in the vector space.

This script is specifically written to help you interact with embedding model from OpenAI. 

For information, check this [documentation](https://python.langchain.com/docs/modules/data_connection/text_embedding/).

- run the script: `python create_index.py`
- enter the absolute path of PDF file that you want to create index for