from pydantic import BaseModel, version

def pydantic_to_dict(b:BaseModel):
    if version.VERSION.startswith("1"):
        return b.dict()
    else:
        return b.model_dump()