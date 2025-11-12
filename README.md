# 🏥 Medical Chatbot with RAG using LLMs, LangChain, Pinecone, Flask & AWS

## 📌 Overview

This project is a full-stack AI-powered **Medical Chatbot** built with **Retrieval-Augmented Generation (RAG)**. It provides accurate, contextual, and real-time responses to healthcare-related queries by combining the power of **Large Language Models (LLMs)** with document retrieval via **Pinecone**, orchestrated using **LangChain**, served through a **Flask API**, and deployed on **AWS**.

---

## 🧠 Tech Stack

- **Large Language Models (LLMs)** - OpenAI GPT-3.5/4
- **LangChain** - Framework for managing chains and memory
- **Pinecone** - Vector DB for semantic search
- **RAG (Retrieval-Augmented Generation)** - Enhances LLM output with external context
- **Flask** - Lightweight backend to serve APIs
- **AWS** - EC2 (backend hosting), S3 (file storage), optional Lambda

---

## 🔁 RAG Pipeline Workflow

````mermaid
graph TD
A[User Query] --> B[Flask API]
B --> C[LangChain Pipeline]
C --> D[Embed Query]
D --> E[Pinecone Search]
E --> F[Retrieve Relevant Docs]
F --> G[LLM with Prompt + Context]
G --> H[Final Response]
H --> I[Return to Frontend/UI]


````

	 

## 💡 Features

### 🧠 Retrieval-Augmented Generation (RAG)
Combines LLMs with document retrieval for reliable, grounded responses. Ensures answers are based on actual medical documents, not just model predictions.

### 📚 Custom Medical Knowledge Base
Easily ingest your own medical documents (PDFs, DOCX, or plain text). Documents are embedded and stored in Pinecone for semantic search.

### 💬 Context-Aware Conversations
Supports memory for multi-turn interactions using LangChain's `ConversationBufferMemory`, enabling natural follow-up questions and continued context.

### 🔐 Cloud-Based & Secure


```

# How to run?
### STEPS:

Clone the repository

```bash
git clonehttps://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
````

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up localhost:
```

### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

    #with specific access

    1. EC2 access : It is virtual machine

    2. ECR: Elastic Container registry to save your docker image in aws


    #Description: About the deployment

    1. Build docker image of the source code

    2. Push your docker image to ECR

    3. Launch Your EC2

    4. Pull Your image from ECR in EC2

    5. Lauch your docker image in EC2

    #Policy:

    1. AmazonEC2ContainerRegistryFullAccess

    2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image

    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot

## 4. Create EC2 machine (Ubuntu)

## 5. Open EC2 and Install docker in EC2 Machine:

    #optinal

    sudo apt-get update -y

    sudo apt-get upgrade

    #required

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

# 6. Configure EC2 as self-hosted runner:

    setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- ECR_REPO
- PINECONE_API_KEY
- OPENAI_API_KEY
