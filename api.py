from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from suggestion_state import get_suggestion, clear_suggestion

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/suggestion")
def get_latest_suggestion():
    suggestion, image_b64 = get_suggestion()
    if suggestion:
        clear_suggestion()
        return {
            "available": True,
            "message": suggestion["message"],
            "recommendation": suggestion["recommendation"],
            "image": image_b64
        }
    return {"available": False}


@app.post("/suggestion/clear")
def clear_latest_suggestion():
    clear_suggestion()
    return {"status": "cleared"}
