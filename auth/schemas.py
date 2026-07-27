"""
Pydantic schemas for signup, login, and token responses.
"""
from pydantic import BaseModel, EmailStr
import datetime


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True
