# from fastapi import FastAPI, Request
# from fastapi.responses import FileResponse
# from pathlib import Path
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from starlette.responses import FileResponse
# app = FastAPI()
# # app.mount("/static", StaticFiles(directory="static"), name="static")
# # app.mount("/", StaticFiles(directory="static", html=True), name="static")

# templates = Jinja2Templates(directory="templates")


# @app.get("/", response_class=HTMLResponse)
# async def read_items():
#     html_content = """
#     <html>
#         <head>
#             <title>Some HTML in here</title>

#         </head>
#         <body>
#             <h1>Look ma! HTML!</h1>
#             <img
#                 src="https://img-cdn.pixlr.com/image-generator/history/65bb506dcb310754719cf81f/ede935de-1138-4f66-8ed7-44bd16efc709/medium.webp"
#                 alt=""
#             />
#         </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content, status_code=200)


# @app.get("/get_image")
# async def get_image():
#     image_path = Path(
#         "./Outputs/TOM/002599_1.jpg")
#     if not image_path.is_file():
#         return {"error": "Image not found on the server"}
#     return FileResponse(image_path)


# @app.get("/items/", response_class=HTMLResponse)
# async def read_items():
#     html_content = """
#     <html>
#         <head>
#             <title>Some HTML in here</title>
#         </head>
#         <body>
#             <h1>Look ma! HTML!</h1>
#         </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content, status_code=200)

# # @app.get("/get_image")
# # async def get_image():
# #     image_path = Path("gfglogo.jpg")
# #     if not image_path.is_file():
# #         return {"error": "Image not found on the server"}
# #     return FileResponse(image_path)


# @app.get("/items/{id}", response_class=HTMLResponse)
# async def read_item(request: Request, id: str):
#     return templates.TemplateResponse(
#         request=request, name="index.html", context={"id": id}
#     )

# # uvicorn main:app --reload

# uvicorn.run("my_fastapi_server:app", host='0.0.0.0', port=8127, workers=2)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app import convertBase64
import uvicorn

app = FastAPI()


class Item(BaseModel):
    image: str


@app.post('/')
async def create_item(item: Item):
    print(item)
    return item


@app.get('/result')
async def get_image():
    return convertBase64()

app.mount("/api", app)
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="127.0.0.1",
                port=8000, reload=True, workers=2)
