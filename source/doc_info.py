from pydantic import BaseModel, Field

class DocInfo(BaseModel):

    structured_document: str = Field(..., description = "Contains the document returned in the suggested structure")
    feedback: str = Field(..., description = "Returns the feedback per section")