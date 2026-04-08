# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

This implementation keeps the same end-to-end behavior as the original project
(Flask + LangChain + OpenAI + Pinecone medical RAG chatbot) while using a
slightly different internal code structure for maintainability.

## Project Structure

```text
.
|- app.py                    # Flask runtime entrypoint
|- streamlit_app.py          # Streamlit runtime entrypoint
|- store_index.py            # Vector index build entrypoint
|- src/
|  |- config.py              # .env loading and runtime settings
|  |- helper.py              # PDF loading, filtering, splitting, embeddings
|  |- index_builder.py       # Pinecone index creation + document upsert
|  |- prompt.py              # System prompt for the assistant
|  |- rag_pipeline.py        # Retriever + LLM RAG chain assembly
|  |- webapp.py              # Flask app factory and routes
|- templates/chat.html       # Chat UI
|- static/style.css          # Chat styles
```

# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/sillyfellow21/Medical-RAG-Chatbot.git
```
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

### Run with Streamlit

```bash
streamlit run streamlit_app.py
```

The Streamlit app uses the same backend modules and includes a retrieval-only
fallback if OpenAI generation is unavailable.

### Streamlit Cloud Setup (Important)

1. Set Main file path to `streamlit_app.py` (not `app.py`).
2. This repo includes `runtime.txt` with `python-3.11` to avoid Cloud runtime
	incompatibilities with LangChain/Pydantic.
3. Add secrets in Streamlit Cloud settings:

```toml
PINECONE_API_KEY="your_key"
OPENAI_API_KEY="your_key"
```

4. Reboot the app after changing Main file path or secrets.

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
## Acknowledgments and Credits

This project was built upon the foundational architecture and codebase provided by [entbappy](https://github.com/entbappy). 

* **Original Project:** [Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS](https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS)
* Modifications made in this version include:
  * Swapped original APIs for [insert your API choices].
  * Refactored frontend and backend logic.
  * [Add any other major changes you made].

A huge thank you to the original author for providing a great starting point!
