from fastapi import FastAPI

app = FastAPI(title="Myanmar Movie Recap AI")


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Myanmar Movie Recap AI"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
