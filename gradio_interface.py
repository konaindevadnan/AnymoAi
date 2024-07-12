import gradio as gr
import requests

def call_fastapi(file):
    url = "http://localhost:8000/uploadfile/"
    files = {'file': file}
    response = requests.post(url, files=files)
    return response.json()["model_url"]

iface = gr.Interface(
    fn=call_fastapi,
    inputs=gr.inputs.File(label="Upload Image"),
    outputs=gr.outputs.Textbox(label="Model URL")
)

iface.launch(server_name="0.0.0.0", server_port=43839)
