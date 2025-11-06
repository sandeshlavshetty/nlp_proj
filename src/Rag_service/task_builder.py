# import ollama
import os
from urllib import response
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any
import json
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import re
import ast
import base64
import os
from typing import Optional
from PIL import Image  # optional: for resizing (pip install pillow)

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
            
    @staticmethod
    def try_parse_json(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # try removing leading/trailing quotes
            s2 = s.strip()
            if (s2.startswith('"') and s2.endswith('"')) or (s2.startswith("'") and s2.endswith("'")):
                s2 = s2[1:-1]
            # try basic unescape of newlines etc
            try:
                return json.loads(s2)
            except json.JSONDecodeError:
                # last resort: try ast.literal_eval (can parse Python dicts)
                try:
                    return ast.literal_eval(s2)
                except Exception:
                    return None

    def _upload_image_to_url(image_path: str) -> Optional[str]:
        """
        OPTIONAL: implement an uploader that returns a public URL (S3, file.io, etc).
        Return None if you don't want to upload.
        """
        return None

    def describe_image(self, image_path: str):
        prompt = "Describe this image in detail. If text is present, extract it."
        system_msg = SystemMessage("You are an image caption and OCR assistant.")
        human_msg = HumanMessage(prompt)

        messages = [system_msg, human_msg]

        # 1) Try provider-supported image kwarg (preferred)
        try:
            # Many LangChain provider wrappers accept an images kwarg (list of paths or bytes).
            # Use images=[image_path] for local file path (works for some local providers).
            response = self.llm.invoke(messages, images=[image_path])
            return response.content
        except TypeError:
            # provider.invoke doesn't accept images kwarg
            pass
        except Exception:
            # provider may raise if local paths are not supported by remote service
            pass

        # 2) Try using a public URL (if you implement uploader)
        url = self._upload_image_to_url(image_path)
        if url:
            human_with_url = HumanMessage(prompt + f"\n\nImage URL: {url}")
            try:
                response = self.llm.invoke([system_msg, human_with_url])
                return response.content
            except Exception:
                pass

        # 3) Fallback: inline base64 data URI inside the message
        #    Resize first if image is large to avoid request size limits
        try:
            # Optional: reduce size if pillow is available and image is large
            try:
                with Image.open(image_path) as im:
                    max_pixels = 1024 * 1024 * 2  # ~2MP
                    if im.width * im.height > max_pixels:
                        ratio = (max_pixels / (im.width * im.height)) ** 0.5
                        new_size = (int(im.width * ratio), int(im.height * ratio))
                        im = im.resize(new_size)
                        # write to bytes
                        from io import BytesIO
                        buf = BytesIO()
                        im.save(buf, format="JPEG", quality=85)
                        img_bytes = buf.getvalue()
                    else:
                        with open(image_path, "rb") as f:
                            img_bytes = f.read()
            except Exception:
                # pillow not available or resize failed, fall back to reading raw bytes
                with open(image_path, "rb") as f:
                    img_bytes = f.read()

            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "jpg"
            data_uri = f"data:image/{ext};base64,{img_b64}"
            human_b64 = HumanMessage(prompt + "\n\nImage (base64): " + data_uri)
            response = self.llm.invoke([system_msg, human_b64])
            return response.content
        except Exception as e:
            print(f"[ERROR] Failed to send image to LLM: {e}")
            return None
    
    def tagger(self, text: str) -> Dict:
        # Construct a rich prompt to extract all necessary info
        prompt = f"""
    You are an intelligent exam paper parser.

    Extract the following information from the given question paper text and return it in **strict JSON** format:

    Fields required:
    1. "date_of_exam" - in format DD/MM/YY (for example: "25/08/22")
    2. "type_of_exam" - one of ["sessional1", "sessional2", "Midsem", "Endsem"]
    3. "paper_code" - example: "CSL420", "HUL322", etc.
    4. "subject_name" - full subject name written after the paper code
    5. "questions" - dictionary where:
    - Keys are question numbers ("Q1", "Q2", etc.)
    - If sub-questions exist (like "A)" or "B)"), their keys should be "Q1_A)", "Q1_B)", etc.
    - Values should be the full question text (without marks or CO codes).

    Example output format:
    {{
    "date_of_exam": "25/08/22",
    "type_of_exam": "Midsem",
    "paper_code": "CSL420",
    "subject_name": "Computer Networks",
    "questions": {{
        "Q1": "Consider a network connected two systems located 9000 kilometers apart...",
        "Q1_A)": "Explain the calculation of minimum sequence number field.",
        "Q2": "How does the stop and wait protocol handle the following..."
    }}
    }}

    Now extract this information from the following text:

    {text}
    """

        # # Send to LLM
        # response = self.llm.chat(
        #     model=self.llm_model,
        #     messages=[
        #         {"role": "system", "content": "You are a text tagger assistant that returns clean, strict JSON only."},
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        system_msg = SystemMessage("You are a text tagger assistant that returns clean, strict JSON only.")
        human_msg = HumanMessage(prompt)

        # Use with chat models
        messages = [system_msg, human_msg]
        response = self.llm.invoke(messages)  # Returns AIMessage
        print(f"[DEBUG] LLM response: {response.content}")
        # Try to parse the LLM output into JSON
        # after printing debug:
        raw = response.content

        # 1) try to extract content inside a fenced code block first
        # extract JSON inside fenced code block if present
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
        if m:
            candidate = m.group(1).strip()
        else:
            # no fenced block found: strip leading/trailing backticks and whitespace
            candidate = raw.strip().strip("`").strip()
        tag_scores = self.try_parse_json(candidate)
        if tag_scores is None:
            print("[ERROR] Failed to parse LLM response as JSON. Returning empty dict.")
            print("[DEBUG] Raw response:\n", raw)
            return {}
        return tag_scores
