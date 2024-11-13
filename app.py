from flask import Flask, render_template, request, jsonify, url_for
import os
import requests
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
import base64
import logging
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# HF_TOKEN = "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"
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(token=os.getenv("HF_TOKEN"))
HF_TOKEN = os.getenv("HF_TOKEN")
def llmfunc(topic, platform, word_count):
    # os.environ['HF_TOKEN'] = HF_TOKEN  
    repo_id = REPO_ID  
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=word_count, temperature=0.7, token=HF_TOKEN)

    template = """Write a blog on the {topic} to be published on {platform} for a total of {word_count} words. Please end the blog with a full stop after including necessary hashtags."""
    prompt = PromptTemplate(template=template, input_variables=['topic', 'platform', 'word_count'])
    model = LLMChain(llm=llm, prompt=prompt)

    result = model.run({"topic": topic, "platform": platform, "word_count": word_count})
    return result

def generate_image(prompt):
    """Generates image based on a text prompt and returns base64 string."""
    client = InferenceClient(token=HF_TOKEN)
    # Generate the image
    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev")

    # Convert JpegImageFile to bytes if needed
    if isinstance(image, Image.Image):
        # Create a BytesIO buffer to hold the image data
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
    else:
        raise TypeError("Expected image to be of type 'PIL.Image.Image'")

    # Encode image bytes to base64
    image_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"

def fetch_web_content(url):
    """Fetches and returns the textual content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching the content: {e}"

def answer_question(url, question):
    """Fetches content from the URL and answers a question based on that content."""
    content = fetch_web_content(url)
    
    if "An error occurred" in content:
        return content
    
    prompt = f"Based on the following content, answer the question and end with a full stop: '{question}'\n\nContent: {content[:1000]}..."
    
    response = client.text_generation(prompt, model=REPO_ID)
    
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    blog_content = None
    image_url = None
    error = None
    url = "https://www.agilisium.com/"  # Hardcoded URL

    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            question = data.get("question")
            logging.debug(f"Received question: {question}")
            if question:
                try:
                    answer = answer_question(url, question)
                    logging.debug(f"Generated answer: {answer}")
                    return jsonify({"answer": answer})
                except Exception as e:
                    logging.error(f"Error answering question: {e}")
                    return jsonify({"error": str(e)}), 500
            else:
                return jsonify({"error": "Please enter a question."}), 400
        else:
            # Handle blog generation and image generation
            if "generate_blog" in request.form:
                topic = request.form.get("topic")
                platform = request.form.get("platform")
                word_count = request.form.get("word_count")
                if topic and platform and word_count:
                    try:
                        word_count = int(word_count)
                        blog_content = llmfunc(topic, platform, word_count)
                        image_prompt = f"An artistic image of {topic} related to {platform}"
                        image_url = generate_image(image_prompt)  # Generate image based on the blog topic
                    except Exception as e:
                        error = f"An error occurred: {str(e)}"
                else:
                    error = "Please enter all required fields."

    return render_template("index.html", blog_content=blog_content, image_url=image_url, error=error)
@app.route("/chatbot", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    question = data.get("question")
    url = "https://www.agilisium.com/"  # This URL can be modified

    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        answer = answer_question(url, question)
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error processing chatbot question: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5500)
