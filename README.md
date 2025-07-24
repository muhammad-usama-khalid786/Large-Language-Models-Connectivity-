# LangChain Model Connectivity Examples

This repository contains examples demonstrating how to connect and use various AI models through LangChain wrappers. Each example shows different approaches for integrating with popular AI providers and open-source models.

## Table of Contents
- [OpenAI Models](#openai-models)
- [Anthropic Models](#anthropic-models)
- [Google Gemini Models](#google-gemini-models)
- [HuggingFace Models](#huggingface-models)
- [Embedding Models](#embedding-models)
- [Setup Instructions](#setup-instructions)

## OpenAI Models

### Chat Models
```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.7, max_tokens=150)
result = model.invoke("Write a poem about technology")
print(result.content)
```

### Legacy Completion Models
```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')
result = llm.invoke("What is the capital of India")
print(result)
```

### Embedding Models
```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1536)
result = embedding.embed_query("Delhi is the capital of India")
```

## Anthropic Models

```python
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20240620')
result = model.invoke('What is the capital of India')
print(result.content)
```

## Google Gemini Models

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
result = model.invoke('What is the capital of India')
print(result.content)
```

## HuggingFace Models

### API-based Inference
```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India")
print(result.content)
```

### Local Pipeline Inference
```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India")
print(result.content)
```

### HuggingFace Embeddings
```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]
vector = embedding.embed_documents(documents)
```

## Embedding Models Comparison

### OpenAI Embeddings
```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills."
]

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query('tell me about dhoni')
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
```

## Setup Instructions

### 1. Environment Setup
Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### 2. Installation
```bash
pip install langchain-openai langchain-anthropic langchain-google-genai langchain-huggingface python-dotenv scikit-learn
```

### 3. Required Dependencies
- `langchain-openai` - For OpenAI models
- `langchain-anthropic` - For Anthropic models
- `langchain-google-genai` - For Google Gemini models
- `langchain-huggingface` - For HuggingFace models
- `python-dotenv` - For environment variable management
- `scikit-learn` - For similarity calculations
- `numpy` - For numerical operations

## Best Practices

### Error Handling
Always wrap API calls in try-except blocks:
```python
try:
    result = model.invoke("Your query here")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")
```

### Model Parameters
- **Temperature**: 0.0 (deterministic) to 2.0 (creative)
- **Max Tokens**: Limit response length
- **Dimensions**: For embeddings, balance between performance and quality

### Model Selection Guide
- **OpenAI**: Best quality, paid API
- **Anthropic**: Strong reasoning capabilities
- **Google Gemini**: Good balance of quality and features
- **HuggingFace API**: Cost-effective with various models
- **HuggingFace Local**: Free but requires local resources

## Common Issues and Solutions

### 1. API Key Errors
- Ensure `.env` file is in project root
- Verify API keys are correct
- Check environment variable names

### 2. Model Loading Issues
- Check internet connectivity for API-based models
- Ensure sufficient disk space for local models
- Verify model names are correct

### 3. Performance Optimization
- Use appropriate dimension limits for embeddings
- Set timeouts for slow inference models
- Utilize GPU when available for local models

## License
This project is for educational purposes. Ensure compliance with respective model providers' terms of service.

## Contributing
Feel free to add more examples or improve existing ones. Please follow the existing code structure and documentation style.
