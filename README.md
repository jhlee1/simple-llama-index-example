# Simple example of building RAG using llamaindex

## Setup

### 1. Download and install [ollama](https://ollama.com/)

### 2. Run the model. In this project, I am using deepseek-R1.

```bash
ollama pull deepseek-r1
```

### 3. Download python3 (I am currently using 3.13.2)

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Optionally, you can run with `venv`

```bash
python3 -m venv venv
source ./venv/bin/activate
```

## How to run

1. build the RAG

```bash
python3 build_rag.py
```

2. Run query

```bash
python3 query_rag.py
```
