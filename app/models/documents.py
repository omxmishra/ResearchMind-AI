from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: Optional[str] = None
    published_date: Optional[date] = None
    updated_date: Optional[date] = None
    arxiv_url: Optional[str] = None
    pdf_url: Optional[str] = None
    word_count: Optional[int] = None
    abstract_length: Optional[int] = None

    @property
    def text_for_embedding(self) -> str:
        return f"{self.title}. {self.title}. {self.abstract}"

    @property
    def short_abstract(self) -> str:
        return self.abstract[:300] + "..." if len(self.abstract) > 300 else self.abstract

    class Config:
        from_attributes = True