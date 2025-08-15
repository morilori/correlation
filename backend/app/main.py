from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .attention import router as attention_router
from .correlation import router as correlation_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(attention_router)
app.include_router(correlation_router)
