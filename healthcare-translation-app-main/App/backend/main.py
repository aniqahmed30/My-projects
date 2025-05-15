import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import openai.error  # For specific OpenAI exceptions

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Ensure the API key is set
if not OPENAI_API_KEY:
    raise RuntimeError("⚠️ Missing API key! Please set OPENAI_API_KEY in the .env file.")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

app = FastAPI()

# Add middleware to handle CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Define a request body model
class TranslationRequest(BaseModel):
    text: str
    target_language: str

@app.get("/")
def home():
    return {"message": "Healthcare Translation API with AI is running!"}

@app.post("/translate")
def translate_text(request: TranslationRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        # Request OpenAI to translate the text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical translation AI. Ensure accurate medical terminology in translations."},
                {"role": "user", "content": f"Translate this text to {request.target_language}: {request.text}"}
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        translated_text = response["choices"][0]["message"]["content"].strip()

        return {"translated_text": translated_text}

    except openai.error.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI quota exceeded. Please check your usage or billing.")
    except openai.error.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key. Check your .env file.")
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
