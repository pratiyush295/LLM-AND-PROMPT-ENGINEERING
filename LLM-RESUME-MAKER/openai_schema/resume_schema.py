from pydantic import BaseModel, HttpUrl, EmailStr
from typing import List, Optional

class ContactInfo(BaseModel):
    name: str
    location: str
    phone: str
    email: str
    github: Optional[str]
    linkedin: Optional[str]
    personal_website: Optional[str]

class Education(BaseModel):
    institution: str
    degree: str
    graduation_date: str
    gpa: Optional[float]
    coursework: List[str]

class Experience(BaseModel):
    company: str
    position: str
    location: str
    start_date: str
    end_date: Optional[str]
    description: List[str]

class Project(BaseModel):
    name: str
    link: Optional[str]
    description: List[str]
    tech_stack: List[str]
    achievements: Optional[List[str]]

class SkillSet(BaseModel):
    programming_languages: List[str]
    technologies: List[str]
    databases: List[str]

class AdditionalInfo(BaseModel):
    awards: List[str]
    certifications: List[str]
    extracurricular: List[str]
    languages: List[str]

class ResumeSchema(BaseModel):
    contact_info: ContactInfo
    education: List[Education]
    professional_experience: List[Experience]
    projects: List[Project]
    skills: SkillSet
    additional_info: Optional[AdditionalInfo]
