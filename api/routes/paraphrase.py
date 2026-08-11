from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from rewriter.rewrite_chain import paraphrase_text
from auth.dependencies import get_current_user
from db.models import User

router = APIRouter()


class ParaphraseRequest(BaseModel):
    text: str
    tone: str = "academic"


class ParaphraseResponse(BaseModel):
    paraphrased_text: str
    tone: str
    original_word_count: int
    paraphrased_word_count: int


@router.post("/paraphrase", response_model=ParaphraseResponse)
def handle_paraphrase(
    req: ParaphraseRequest,
    current_user: User = Depends(get_current_user),
):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        result = paraphrase_text(req.text, req.tone)
        orig_words = len(req.text.strip().split())
        new_words = len(result.strip().split())
        return ParaphraseResponse(
            paraphrased_text=result,
            tone=req.tone,
            original_word_count=orig_words,
            paraphrased_word_count=new_words,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
