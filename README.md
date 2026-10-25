# 🤖 Agentic Chatbot

An agentic AI chatbot built with **Streamlit, LangGraph, LangChain, and Azure OpenAI**. It uses external tools to answer questions, search the web, analyze uploaded PDFs, perform calculations, retrieve weather information and stock prices, and request human approval for simulated stock purchases.

The application also supports conversational memory, streaming responses, and Docker-based deployment with a GitHub Actions CI/CD workflow for AWS EC2.

## ✨ Features

- **AI-Powered Conversations:** Uses Azure OpenAI for intelligent, context-aware responses.
- **Conversational Memory:** Maintains conversation history using LangGraph checkpoints and SQLite.
- **PDF Question Answering:** Upload PDFs and query their contents using Retrieval-Augmented Generation (RAG).
- **Vector Search:** Uses Hugging Face embeddings and FAISS for document retrieval.
- **Web Search:** Retrieves relevant information from the web using Tavily.
- **Calculator:** Performs mathematical calculations through an integrated tool.
- **Weather Information:** Retrieves current weather data using Weatherstack.
- **Stock Market Data:** Fetches stock prices using Alpha Vantage.
- **Human-in-the-Loop:** Requests human approval before executing simulated stock purchases.
- **Streaming Responses:** Displays AI-generated responses progressively in the Streamlit interface.
- **Docker Support:** Containerizes the application for consistent deployment.
- **CI/CD Pipeline:** Automates Docker image builds and deployment to AWS EC2 using GitHub Actions.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Interactive chatbot interface |
| LangChain | LLM integration and tool development |
| LangGraph | Agent workflow orchestration and conversation state management |
| Azure OpenAI | Language model inference |
| Hugging Face | Text embeddings |
| FAISS | Vector storage and similarity search |
| PyPDF | PDF text extraction |
| Tavily | Web search |
| Weatherstack | Weather information |
| Alpha Vantage | Stock market data |
| SQLite | Persistent conversation checkpoints |
| Docker | Application containerization |
| GitHub Actions | CI/CD automation |
| AWS EC2 | Deployment infrastructure |
| LangSmith | Optional tracing and observability |

## 🏗️ Project Architecture

```text
                         ┌───────────────────┐
                         │       User        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │    Streamlit UI   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ LangGraph Workflow│
                         └─────────┬─────────┘
                                   │
                   ┌───────────────┼────────────────┐
                   │               │                │
                   ▼               ▼                ▼
            ┌────────────┐  ┌────────────┐  ┌──────────────┐
            │ Azure      │  │ External   │  │ PDF Retrieval│
            │ OpenAI     │  │ Tools      │  │ with FAISS   │
            └────────────┘  └────────────┘  └──────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
               Web Search      Weather      Stock Prices
                    │
                    ▼
                Calculator

                         ┌───────────────────┐
                         │ SQLite Checkpoints│
                         └───────────────────┘
```

### How It Works

1. The user interacts with the chatbot through the Streamlit interface.
2. LangGraph manages the agent workflow, conversation state, and tool execution.
3. Azure OpenAI processes user queries and determines when external tools are needed.
4. The agent invokes the appropriate tool to retrieve information or perform a calculation.
5. For PDF-related queries, the RAG pipeline retrieves relevant document chunks from the FAISS index.
6. LangGraph checkpoints preserve conversation state in the SQLite database.
7. The application streams the response back to the user.

## 📂 Project Structure

```text
Agentic-Chatbot/
│
├── .github/
│   └── workflows/
│       └── cicd.yaml             # GitHub Actions CI/CD workflow
│
├── .vscode/
│   └── settings.json             # VS Code settings
│
├── app.py                         # Streamlit application
├── chatbot.py                     # LangGraph agent and tools
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image configuration
├── testing.ipynb                  # Exploratory development notebook
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git exclusions
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

**Runtime files generated by the application:**

- `Chat.db` — Stores conversation checkpoints.
- `faissDB/` — Stores the FAISS vector index generated from uploaded PDFs.

These files may be created at runtime and are not necessarily included in the repository.

## ⚙️ Requirements

Before running the application, ensure you have:

- Python 3.11 or later
- pip
- An Azure OpenAI resource with a deployed language model
- A Tavily API key for web search
- A Weatherstack API key for weather information
- An Alpha Vantage API key for stock prices
- Docker, if you want to run the application in a container

**Optional:** LangSmith credentials for tracing and observability.

## 🔑 Environment Variables

Create a `.env` file in the project root and configure the required credentials.

```ini
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name

# Tavily Web Search
TAVILY_API_KEY=your-tavily-api-key

# Weatherstack
WEATHERSTACK_API_KEY=your-weatherstack-api-key

# LangSmith (Optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=agentic-chatbot
```

Replace the placeholder values with your actual credentials.

> **Important:** The Azure OpenAI configuration must be available when the application initializes. Other credentials are required when their corresponding tools are used.

Never commit your `.env` file or expose API keys in your source code.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a Virtual Environment

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create the `.env` file described in the [Environment Variables](#-environment-variables) section and add your API credentials.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

The chatbot should now be accessible through the Streamlit interface.

## 📄 PDF Question Answering

The application supports document-based question answering using Retrieval-Augmented Generation (RAG).

### Workflow

1. Upload a PDF through the Streamlit interface.
2. Extract the document text using PyPDF.
3. Split the extracted text into smaller chunks.
4. Generate embeddings using a Hugging Face embedding model.
5. Store the embeddings in a FAISS vector index.
6. Retrieve relevant chunks when the user asks a question.
7. Use the retrieved context to generate an answer with Azure OpenAI.

### Storage

The generated vector index is stored locally in:

```text
faissDB/
```

**Current limitation:** The implementation processes the first uploaded PDF and replaces the existing local FAISS index when another PDF is uploaded. Multiple-document management and persistent remote vector storage are not currently configured.

## 🧰 Available Tools

The agent can use the following tools depending on the user's query.

| Tool | Description |
|---|---|
| `rag_tool` | Retrieves relevant information from uploaded PDF documents. |
| `search_tool` | Searches the web using Tavily. |
| `calculator` | Evaluates supported mathematical expressions. |
| `fetch_weather` | Retrieves current weather information through Weatherstack. |
| `get_stock_price` | Retrieves stock price data through Alpha Vantage. |
| `purchase_stock` | Simulates a stock purchase after human approval. |

> **Disclaimer:** The stock purchase functionality is a simulation and does not place real brokerage orders or execute actual financial transactions.

## 🐳 Running with Docker

Docker allows you to run the chatbot in a containerized environment.

### 1. Build the Docker Image

```bash
docker build -t agentic-chatbot .
```

### 2. Run the Container

```bash
docker run --rm \
  --name agentic-chatbot \
  --env-file .env \
  -p 8501:8501 \
  agentic-chatbot
```

The command above uses Unix-style line continuation. On Windows PowerShell, use:

```powershell
docker run --rm `
  --name agentic-chatbot `
  --env-file .env `
  -p 8501:8501 `
  agentic-chatbot
```

### 3. Access the Application

Open:

```text
http://localhost:8501
```

The application runs inside the container and exposes port `8501`.

> **Note:** Ensure that the Dockerfile configures Streamlit to listen on `0.0.0.0` and port `8501`.

## 💾 Data Persistence

The application uses local storage for conversation checkpoints and PDF embeddings.

| File or Directory | Purpose |
|---|---|
| `Chat.db` | SQLite database containing conversation checkpoints |
| `faissDB/` | FAISS vector index for uploaded PDF documents |

### Deployment Considerations

The deployment workflow replaces the existing Docker container. Without Docker volumes or external storage, local conversation data and PDF indexes may be lost when the container is recreated.

For persistent production deployments, consider:

- Docker volumes for persistent local storage.
- Managed databases or durable storage for conversation data.
- Object storage for uploaded PDF documents.
- Persistent vector storage for document embeddings.

## 🔄 GitHub Actions CI/CD

The CI/CD workflow is defined in:

```text
.github/workflows/cicd.yaml
```

It is configured to run when code is pushed to the `main` branch or when the workflow is triggered manually through GitHub Actions.

### Deployment Workflow

1. Check out the repository.
2. Execute the configured validation and integration steps.
3. Build the Docker image.
4. Push the image to Docker Hub.
5. Execute deployment commands on a configured AWS EC2 self-hosted runner.
6. Pull the latest Docker image.
7. Stop and replace the existing application container.
8. Start the updated Streamlit application on port `8501`.
9. Verify container status and the Streamlit health endpoint.

### Deployment Configuration

| Setting | Value |
|---|---|
| Deployment platform | AWS EC2 |
| CI/CD platform | GitHub Actions |
| Container registry | Docker Hub |
| Container name | `stapp` |
| Application port | `8501` |
| Health endpoint | `http://localhost:8501/_stcore/health` |

The AWS EC2 instance must be configured as a GitHub self-hosted runner with Docker installed and the necessary permissions to execute deployment commands.

### Required GitHub Secrets

Configure the following repository secrets under **Settings → Secrets and variables → Actions**.

**Docker Hub**

```text
DOCKER_USERNAME
DOCKER_PASSWORD
IMAGE_NAME
```

**Azure OpenAI**

```text
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT_NAME
```

**External Tools**

```text
TAVILY_API_KEY
WEATHERSTACK_API_KEY
```

**LangSmith**

```text
LANGSMITH_TRACING
LANGSMITH_ENDPOINT
LANGSMITH_API_KEY
LANGSMITH_PROJECT
```

Ensure that the workflow passes the required variables to the container when starting it.

> **Note:** The workflow currently contains placeholder linting and unit-testing steps. A complete automated test suite is not included in the current project.

## 🧪 Testing

The repository includes an exploratory Jupyter notebook:

```text
testing.ipynb
```

The notebook was used during development and references a local PDF file that is not included in the repository.

Automated testing should be added to validate:

- Agent workflow execution.
- Tool invocation and error handling.
- PDF ingestion and retrieval.
- Conversation checkpoint persistence.
- Environment variable configuration.
- Docker health and application startup.

## 🔐 Security Considerations

Follow these recommendations before deploying the application publicly.

- **Protect API keys:** Never commit `.env` files, credentials, or secrets to GitHub.
- **Secure Docker builds:** Add `.env` and other sensitive files to `.dockerignore`.
- **Remove embedded credentials:** The current stock price tool contains an embedded Alpha Vantage credential in `chatbot.py`. Move it to an environment variable and rotate the credential if it has been exposed.
- **Validate PDF uploads:** Restrict file types and sizes and handle malformed documents safely.
- **Use trusted FAISS indexes:** FAISS loading uses deserialization functionality that can be unsafe with untrusted index files. Load only trusted data.
- **Persist important data:** Configure durable storage if conversations and uploaded documents must survive deployments.
- **Protect the deployment environment:** Restrict access to the EC2 instance, GitHub secrets, and Docker commands.
- **Avoid exposing sensitive information:** Do not display API keys or internal exception details to end users.
- **Review human approval flows:** Ensure approval is required before any sensitive action is performed.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.