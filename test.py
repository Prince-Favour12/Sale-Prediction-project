from fastapi import FastAPI, HTTPException, status, Path
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

users = {
    1 : {
        "name": "Josh",
        "website": "www.zerotoknowing.com",
        "age": 28,
        "role": "developer"
    },
    2 : {
        "name": "prince",
        "website": "www.contentbridge.com",
        "age": 18,
        "role": "developer"
    }
}

@app.get("/")
def root():
    return {"message": "welcome to your introduction to FastAPI"}


# get users
@app.get("/users/{user_id}")
def get_user(user_id: int = Path(..., description="The ID you want to get", gt= 0, lt= 100)):
    if user_id not in users:
        raise HTTPException(status_code= 404, detail= "User not found!")
    return users[user_id]
