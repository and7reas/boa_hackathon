from pydantic import BaseModel, Field

class DocInfo(BaseModel):

    standardized_document: str = Field(..., description = "Contains the document returned in the suggested structure")
    missing_information_feedback: str = Field(..., description = "Returns the feedback per section")