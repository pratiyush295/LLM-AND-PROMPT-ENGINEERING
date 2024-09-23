from pydantic import BaseModel, EmailStr, HttpUrl
from typing import List, Optional

class Contact(BaseModel):
    name: str
    address: str
    mobile: str
    email: str
    linkedin: str

class TopSkills(BaseModel):
    skills: List[str]

class Summary(BaseModel):
    text: str

class ExperienceItem(BaseModel):
    company: str
    position: str
    duration: str
    location: str
    description: str

class EducationItem(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    duration: str

class LinkedinResumeSchema(BaseModel):
    contact: Contact
    top_skills: TopSkills
    summary: Summary
    experience: List[ExperienceItem]
    education: List[EducationItem]
