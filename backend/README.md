# BERT Attention Analysis Backend

FastAPI backend for BERT attention visualization. Provides a single endpoint to analyze text and return attention matrices using BERT-base-uncased.

## API Endpoint

- `POST /attention` - Analyzes input text and returns tokenized words with attention matrix
- Automatic documentation available at `/docs` when running

## Dependencies

- FastAPI for web framework
- PyTorch and Transformers for BERT model
- uvicorn for ASGI server
