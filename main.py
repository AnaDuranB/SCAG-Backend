from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from question_generator import QuestionGenerator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

try:
    generator = QuestionGenerator()
except Exception as e:
    raise RuntimeError(f"Error al inicializar el generador de preguntas: {str(e)}")

@app.post("/generate")
async def generate_question_answer(request: TextRequest):
    """
    Endpoint para generar preguntas y respuestas a partir del texto proporcionado.
    """
    try:
        processed_questions = generator.process_text(request.text)

        if processed_questions:
            for q in processed_questions:
                if not q.get('question') or not q.get('options') or not q.get('correct_answer'):
                    raise HTTPException(status_code=500, detail="Faltan campos en una o más preguntas generadas.")

            return {
                "questions": processed_questions
            }
        else:
            raise HTTPException(status_code=400, detail="No se pudo generar ninguna pregunta.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API de generación de preguntas está activa y funcionando correctamente."}
