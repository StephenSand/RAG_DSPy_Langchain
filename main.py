from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from typing import Annotated
from pydantic import BaseModel
import asyncio
from rag import main, reset_chroma

app = FastAPI()


def home_html():
    html_content = """
        <!DOCTYPE html>
        <head>
            <style>
                body {background-color: #2D3748;}
                a {
                  color: CornFlowerBlue;
                  text-decoration: none;
                  border: 4px solid SlateGrey;
                  padding: 10px 20px;
                }
                .content {
                  margin: auto;
                  padding: 10px;
                  color: CornFlowerBlue;
                  text-align: center;
                  font-family: "Lucida Console", "Courier New", monospace;
                }
            </style>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Home</title>
        </head>
        <body>
            <div class="content">
                <h1>Welcome.</h1>
                <p>This is a lightweight RAG application designed to read company products and services pages.</p>
                <p>It uses FastApi, ChromaDB, DSPy, Langchain, and qwen2.5:0.5b through Ollama.</p>
                <br>
                <p>Simply enter the url and your question as a command, wait a little, and you will get responses from both the DSPy and Langchain agents.</p>
                <p>Secondarily, you will be able to compare the DSPy and Langchain agents, as their reponses differ.</p>
                <p>To continue please click on the Form button.</p>
                <br>
                <a href="http://127.0.0.1:80/initial_form/">Form</a>
            </div>
        </body>
        </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


def initial_form_html():
    html_content = """
        <!DOCTYPE html>
        <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {background-color: #2D3748;}
                    .content {
                      margin: auto;
                      padding: 10px;
                      color: CornFlowerBlue;
                      text-align: center;
                      font-family: "Lucida Console", "Courier New", monospace;
                    }
                    .loader {
                      border: 16px solid #f3f3f3;
                      border-radius: 50%;
                      border-top: 16px solid blue;
                      border-bottom: 16px solid blue;
                      width: 120px;
                      height: 120px;
                      -webkit-animation: spin 2s linear infinite;
                      animation: spin 2s linear infinite;
                      position: relative;
                      left: 46%;
                      top: 50%;
                    }

                    @-webkit-keyframes spin {
                      0% { -webkit-transform: rotate(0deg); }
                      100% { -webkit-transform: rotate(360deg); }
                    }

                    @keyframes spin {
                      0% { transform: rotate(0deg); }
                      100% { transform: rotate(360deg); }
                    }
                </style>
                <title>Form</title>
            </head>
            <body>
                <div class="content">
                    <h1>Form.</h1>
                    <form action="http://127.0.0.1:80/post_initial_form/" enctype="application/x-www-form-urlencoded" id="myForm" method="post">
                        <label for="website_input">Website</label>
                        <input type="text" id="website_input" name="website_input" size="42" value="https://set3.com/san-diego-data-center-cleaning"><br>
                        <br>
                        <label for="question_input">Command</label>
                        <input type="text" id="question_input" name="question_input" size="42" value="What services are offered"><br>
                        <br>
                        <button type="button" onclick="submitForm()">Submit</button>
                    </form>
                    <br><br>
                    <div id="loadingDiv"></div>
                </div>
                <script>
                function submitForm(){
                    document.getElementById("loadingDiv").innerHTML = `<h2>Submitted!</h2><p>This may take a while.</p><br><p>First we scrape the url, embed the text, and save it to our chroma database.</p><p>Next we embed the question and run a similarity search against the embedded texts.</p><p>Last we send the most relevant text embeddings to both our DSPy and Langchain agents to respond with an answer.</p><p>Since we are using a locally run model this will take about 6-10 minutes.</p><div class="loader"></div>`
                    document.getElementById("myForm").submit();
                }
                </script>
            </body>
        </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

def response_html(dspy_response, lc_response):
    head_html = """
        <!DOCTYPE html>
        <head>
            <style>
                body {background-color: #2D3748;}
                a {
                  color: CornFlowerBlue;
                  text-decoration: none;
                  border: 4px solid SlateGrey;
                  padding: 10px 20px;
                }
                .content {
                  margin: auto;
                  padding: 10px;
                  color: CornFlowerBlue;
                  font-family: "Lucida Console", "Courier New", monospace;
                }
            </style>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Response</title>
        </head>
    """
    body_html = f"""
        <body>
            <div class="content">
                <h1 style="text-align: center;">Response</h1>
                <br>
                <h2>DSPy</h2>
                <p>First, let's take a look at the DSPy agent's response.</p>
                <code style="color:white;">{dspy_response}</code>
                <p>As you can see, DSPy reponds with a perfect json object.</p>
                <p>Actually, in the console, it responds with a python data type. In particular, this would be a python dictionary with ints as keys and strings as values.</p>
                <p>DSPy can respond with other python data types such as ints, bools, floats, and lists.</p>
                <p>This is revolutionary as these responses can be used in functions without the need to parse strings or json.</p>
                <p>One can also be sure that the response will be the data type required, dashing any doubts of malformed json responses that could break the function.</p>
                <br><br>
                <h2>Langchain</h2>
                <p>Next, let's take a look at the Langchain agent's response.</p>
                <code style="color:white;">{lc_response}</code>
                <p>Like most RAG frameworks, the Langchain agent reponds in markdown.</p>
                <p>This is fine if a simple text response is required, but it would need some parsing to be used further in a function if you plan to use it as something other than a string.</p>
                <p>A constant criticism of Langchain is that it often abstracts too much, potentially providing inaccurate information.</p>
                <p>On the other hand, the Langchain agent does generate longer, more in-depth reports.</p>
                <p></p>
                <br><br>
                <div style="text-align: center;">
                    <a href="http://127.0.0.1:80/secondary_form/">Ask another question</a>
                    <a href="http://127.0.0.1:80/initial_form/">Use a new website</a>
                <div>
            </div>
        </body>
        </html>
    """
    html_content = head_html + body_html
    return HTMLResponse(content=html_content, status_code=200)


def secondary_form_html():
    html_content = """
        <!DOCTYPE html>
        <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {background-color: #2D3748;}
                    .content {
                      margin: auto;
                      padding: 10px;
                      color: CornFlowerBlue;
                      text-align: center;
                      font-family: "Lucida Console", "Courier New", monospace;
                    }
                    .loader {
                      border: 16px solid #f3f3f3;
                      border-radius: 50%;
                      border-top: 16px solid blue;
                      border-bottom: 16px solid blue;
                      width: 120px;
                      height: 120px;
                      -webkit-animation: spin 2s linear infinite;
                      animation: spin 2s linear infinite;
                      position: relative;
                      left: 46%;
                      top: 50%;
                    }

                    @-webkit-keyframes spin {
                      0% { -webkit-transform: rotate(0deg); }
                      100% { -webkit-transform: rotate(360deg); }
                    }

                    @keyframes spin {
                      0% { transform: rotate(0deg); }
                      100% { transform: rotate(360deg); }
                    }
                </style>
                <title>Form</title>
            </head>
            <body>
                <div class="content">
                    <h1>Form.</h1>
                    <form action="http://127.0.0.1:80/post_secondary_form/" enctype="application/x-www-form-urlencoded" id="myForm" method="post">
                        <label for="question_input">Command</label>
                        <input type="text" id="question_input" name="question_input" size="42"><br>
                        <br>
                        <button type="button" onclick="submitForm()">Submit</button>
                    </form>
                    <br><br>
                    <div id="loadingDiv"></div>
                </div>
                <script>
                function submitForm(){
                    document.getElementById("loadingDiv").innerHTML = `<h2>Submitted!</h2><p>This may take a while.</p><br><p>First, we embed the question and run a similarity search against the embedded texts in our chroma database.</p><p>Then, we send the most relevant text embeddings to both our DSPy and Langchain agents to respond with an answer.</p><p>Since we are using a locally run model this will take about 6-10 minutes.</p><div class="loader"></div>`
                    document.getElementById("myForm").submit();
                }
                </script>
            </body>
        </html>
    """
    return HTMLResponse(content=html_content, status_code=200)



@app.get("/", response_class=HTMLResponse)
async def home():
    return home_html()


@app.get("/initial_form/", response_class=HTMLResponse)
async def initial_form():
    await reset_chroma()
    return initial_form_html()


class InitialFormData(BaseModel):
    website_input: str
    question_input: str
    model_config = {"extra": "forbid"}


class SecondaryFormData(BaseModel):
    question_input: str
    model_config = {"extra": "forbid"}


@app.post("/post_initial_form/", response_class=HTMLResponse)
async def post_initial_form(data: Annotated[InitialFormData, Form()]):
    final = await main(data.website_input, data.question_input)
    return response_html(str(final[0]), str(final[1]))

@app.get("/secondary_form/", response_class=HTMLResponse)
async def secondary_form():
    return secondary_form_html()


@app.post("/post_secondary_form/", response_class=HTMLResponse)
async def post_secondary_form(data: Annotated[SecondaryFormData, Form()]):
    final = await main(None, data.question_input)
    return response_html(str(final[0]), str(final[1]))


