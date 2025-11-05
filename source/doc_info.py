from pydantic import BaseModel, Field

class DocInfo(BaseModel):

    standardized_document: dict = Field(..., description = "Contains the document returned in the suggested structure")
    missing_information_feedback: dict = Field(..., description = "Returns the feedback per section")