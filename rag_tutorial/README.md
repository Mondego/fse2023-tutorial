# Retrieval-Augmented Generation Overview
RAG is a technique for augmenting LLM knowledge with additional, often private or real-time, data.

LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on. If you want to build AI applications that can reason about private data or data introduced after a model's cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

source: [langchain](https://python.langchain.com/docs/use_cases/question_answering/)


### create_index.py

Indexing: a pipeline for ingesting data from a source and indexing it. This usually happen offline.

Embeddings create a vector representation of a piece of text. This is useful because it means we can think about text in the vector space, and do things like semantic search where we look for pieces of text that are most similar in the vector space.

This script is specifically written to help you interact with embedding model from OpenAI. For more information, check this [documentation](https://python.langchain.com/docs/modules/data_connection/text_embedding/).

- run 

    ```
    python create_index.py
    ```

### search.py

This folder includes some papers about software architechture. Ask yourself questions below and see if you could get them correctly.

- questions 1

    ```
    Self-protection is generally concerned with? Maintenance, Reliability, Security, or Intellectual Property 
    ```

- questions 2

    ```
    Self-healing is generally concerned with? Intellectual property, Reliability, Maintenance, or Security
    ```

- questions 3

    ```
    Which of the following options best captures the structure of an autonomic element and the sequence with which the various activities are performed? Execute, Detect, Crash, and Restart \n Detect, Execute, Effect, and Report \n Reboot, Reanalyze, Restart, and Report \n Monitor, Analyze, Plan, and Execute 
    ```

- questions 4

    ```
    Which of these patterns is an example of a proactive approach to self-adaptation? Protective Wrapper Pattern, Agreement-based Redundancy, Software Rejuvenation Pattern, None of the options
    ```

- questions 5

    ```
    Which of these is not one of the three layers for self-management? Component Control, Object Management,  Goal Management, Change Management 
    ```

Now try to ask those questions using RAG 

- run

    ```
    python search.py
    ```

- enter the question 
