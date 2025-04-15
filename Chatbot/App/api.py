from fastapi import FastAPI
from pydantic import BaseModel
from functions import *  

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    output = graph.invoke({
        "question": request.question,
        "original_q": request.question,
    })
    return {"output": output}