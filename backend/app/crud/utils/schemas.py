from pydantic import BaseModel, EmailStr,  ConfigDict
from typing import Optional
from datetime import datetime

# Existing schemas
class UserCreate(BaseModel):
    userID: str
    userName: str
    userPasswordHash: str
    userEmail: EmailStr
    userCountry: Optional[str] = None
    userLanguage: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)

class ArticleCreate(BaseModel):
    articleTitle: str
    articleAuthor: str
    content: Optional[str]
    imageURL: Optional[str] = None
    travelCountry: str
    travelCity: str
    shareLink: Optional[str] = None
    price: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)

class ArticleUpdate(BaseModel):
    articleTitle: Optional[str] = None
    content: Optional[str] = None
    imageURL: Optional[str] = None
    travelCountry: Optional[str] = None
    travelCity: Optional[str] = None
    shareLink: Optional[str] = None
    price: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)

class CommentCreate(BaseModel):
    articleID: str
    commentAuthor: str
    content: str
    
    model_config = ConfigDict(from_attributes=True)

class CommentUpdate(BaseModel):
    content: str
    
    model_config = ConfigDict(from_attributes=True)

# New schema for Like
class LikeCreate(BaseModel):
    articleID: str
    userID: str
    
    model_config = ConfigDict(from_attributes=True)

class DailyCreate(BaseModel):
    date: datetime
    season: str
    weather: str
    temperature: float
    mood: Optional[str]
    country: str

    model_config = ConfigDict(from_attributes=True)

class DailyRead(DailyCreate):
    id: int
    createdAt: datetime
    
