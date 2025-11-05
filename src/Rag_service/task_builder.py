# import ollama
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
class LLMtool:
    def __init__(self, llm_model: str = "gemma2-9b-it"):
        self.llm_model = llm_model
        print(f"[INFO] LLM initialized: {llm_model}")
        if "GROQ_API_KEY" in os.environ:
            groq_api_key = os.environ["GROQ_API_KEY"]
            self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
            print(f"[INFO] Groq LLM initialized: {llm_model}")
        elif "GOOGLE_API_KEY" in os.environ:
            google_api_key = os.environ["GOOGLE_API_KEY"]
            self.llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=llm_model)
            print(f"[INFO] Google LLM initialized: {llm_model}")

    def describe_image(self, image_path):
        prompt = "Describe this image in detail. If text is present, extract it."
        response = self.llm.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are an image caption and OCR assistant."},
                {"role": "user", "content": prompt, "images": [image_path]}
            ]
        )
        return response["message"]["content"]