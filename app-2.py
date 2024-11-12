import gradio as gr
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def extract_text_from_pdf(pdf_path):
    # Open the provided PDF file
    doc = fitz.open(pdf_path)
    text = ""

    # Extract text from each page
    for page in doc:
        text += page.get_text()

    doc.close()  # Ensure the PDF file is closed
    return text


def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        print(f"Token: {token}")  # Debugging statement to trace tokens
        yield response  # Yield the complete response up to this point


def process_resume_and_respond(pdf_file, message, history, system_message, max_tokens, temperature, top_p):
    # Extract text from the PDF file
    resume_text = extract_text_from_pdf(pdf_file.name)
    # Combine the resume text with the user message
    combined_message = f"Resume:\n{resume_text}\n\nUser message:\n{message}"
    # Respond using the combined message
    response_gen = respond(combined_message, history, system_message, max_tokens, temperature, top_p)
    response = "".join([token for token in response_gen])
    return response


# Store the uploaded PDF content globally
uploaded_resume_text = ""


def upload_resume(pdf_file):
    global uploaded_resume_text
    uploaded_resume_text = extract_text_from_pdf(pdf_file.name)
    return "Resume uploaded successfully! now click on chat with job advisor right above this tab to start chatting!"


def respond_with_resume(message, history, system_message, max_tokens, temperature, top_p):
    global uploaded_resume_text
    # Combine the uploaded resume text with the user message
    combined_message = f"Resume:\n{uploaded_resume_text}\n\nUser message:\n{message}"
    # Respond using the combined message
    response_gen = respond(combined_message, history, system_message, max_tokens, temperature, top_p)
    # Collect all tokens generated
    response = ""
    for token in response_gen:
        response = token  # Update the response with the latest token
    return response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
upload_interface = gr.Interface(
    upload_resume,
    inputs=gr.File(label="Upload Resume PDF"),
    outputs=gr.Textbox(label="Upload Status"),
)

chat_interface = gr.ChatInterface(
    respond_with_resume,
    additional_inputs=[
        gr.Textbox(value="You are a Job Advisor Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

demo = gr.TabbedInterface(
    [upload_interface, chat_interface],
    ["Upload Resume", "Chat with Job Advisor"]
)


if __name__ == "__main__":
    demo.launch()
