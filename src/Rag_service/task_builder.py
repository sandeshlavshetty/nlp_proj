# import ollama
import os
from urllib import response
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from google import genai
except Exception:
    genai = None
    print("[WARN] optional package 'google.genai' not importable. GenAI vision features will be disabled until you install 'google-genai'.")
from typing import List, Dict, Any
import json
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import re
import ast
import base64
import os
from typing import Optional
from PIL import Image  # optional: for resizing (pip install pillow)
import fitz  # PyMuPDF
import tempfile
import os
from PIL import Image
import io

load_dotenv()
class LLMtool:
    def __init__(self, llm_model: str = "gemini-2.5-flash"):
        self.llm_model = llm_model
        print(f"[INFO] LLM initialized: {llm_model}")
        if "GROQ_API_KEY" in os.environ:
            groq_api_key = os.environ["GROQ_API_KEY"]
            # self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
            print(f"[INFO] Groq LLM initialized: {llm_model}")
        elif "GOOGLE_API_KEY" in os.environ:
            google_api_key = os.environ["GOOGLE_API_KEY"]
            self.llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=llm_model)
            print(f"[INFO] Google LLM initialized: {llm_model}")
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Environment variable GOOGLE_API_KEY or GEMINI_API_KEY must be set")
            self.client = genai.Client(api_key=api_key)
            print(f"[INFO] Google GenAI client initialized with model {llm_model}")
            
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

    def _upload_image_to_url(self,image_path: str) -> Optional[str]:
        """
        OPTIONAL: implement an uploader that returns a public URL (S3, file.io, etc).
        Return None if you don't want to upload.
        """
        return None

    def ocr_pdf_with_genai(self, uploaded_file, vision_model_name: str = "gemini-2.5-flash") -> str:
        """
        Process a PDF via PyMuPDF, split pages into halves (for clarity),
        then send each image part to the GenAI vision model to extract full text.
        Returns the concatenated extracted text.
        """
        try:
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            doc = fitz.open(tmp_path)
            all_text = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                page_img = Image.open(io.BytesIO(img_data))
                w, h = page_img.size

                # Optionally split image if very tall
                halves = [page_img.crop((0, 0, w, h//2)), page_img.crop((0, h//2, w, h))]

                page_text = ""
                for part_idx, part in enumerate(halves):
                    prompt = (
                        "Extract ALL text from this exam paper image.\n"
                        "- Keep exact wording.\n"
                        "- Maintain numbering.\n"
                        "- Do not summarise or skip anything.\n"
                    )

                    # The GenAI SDK allows multimodal input: text prompt + image as bytes
                    # According to docs: client.models.generate_content(model=..., contents=[prompt, image_bytes]) :contentReference[oaicite:2]{index=2}
                    image_bytes = io.BytesIO()
                    part.save(image_bytes, format="PNG")
                    image_bytes = image_bytes.getvalue()
                    
                    image_part = genai.types.Part.from_bytes(data=image_bytes, mime_type="image/png")


                    response = self.client.models.generate_content(
                        model=vision_model_name,
                        contents=[prompt, image_part]
                    )
                    # get response text
                    text_piece = getattr(response, "text", None) or getattr(response, "content", "")
                    page_text += text_piece + "\n"
                    print(f"[DEBUG] Extracted text from page {page_num+1}, part {part_idx+1}:\n{text_piece}")

                all_text.append(page_text)
            doc.close()
            os.unlink(tmp_path)
            return "\n".join(all_text)
        except Exception as e:
            print(f"[ERROR] Exception during PDF OCR: {e}")
            return ""



    
    def tagger(self, text: str) -> Dict:
        # Construct a rich prompt to extract all necessary info
        prompt = f"""
You are an intelligent exam paper parser.

Your task is to extract structured information from the given question paper text and return it in **strict JSON format only**.

### Required JSON structure:
{{
  "date_of_exam": "DD/MM/YY",
  "type_of_exam": "Midsem" | "Endsem" | "sessional1" | "sessional2",
  "paper_code": "CSL302",
  "subject_name": "Computer Networks",
  "questions": {{
    "Q1_A": "Full question text of Q1(A)",
    "Q1_B": "Full question text of Q1(B)",
    "Q2_A": "Full question text of Q2(A)",
    "Q2_B": "Full question text of Q2(B)",
    ...
  }}
}}

---

### Extraction Rules:

1. **Main questions**
   - Always start with labels like “Q1”, “Q2”, “Q3”, etc.
   - You must capture every question number even if it has only one subpart.

2. **Subquestions**
   - Denoted by capital letters **A, B, C, D, E** in parentheses or after the question number.
   - Use the key format `"Q1_A"`, `"Q1_B"`, etc.
   - Combine everything under that subquestion, including inner parts labeled “i.”, “ii.”, “iii.”, etc., into a single text block.
   - Preserve tables, equations, or diagrams as plain text references (like "[Table shown below]" or "[Graph shown above]").

3. **Inner subpoints (i, ii, iii, etc.)**
   - Do **not** split them into separate questions.
   - Keep them inline as text inside that subquestion value.

4. **Header extraction**
   - Date: extract from “Date:” or similar header lines.
   - Paper code: typically like “CSL302”.
   - Subject name: the text right after the paper code.
   - Type of exam: deduce from “End Semester”, “Mid Semester”, “Sessional I/II”.

5. **Do not include** marks, CO codes, or “Important Instructions”.

---

### Example Output:
{{
  "date_of_exam": "21/11/22",
  "type_of_exam": "Endsem",
  "paper_code": "CSL302",
  "subject_name": "Computer Networks",
  "questions": {{
    "Q1_A": "Suppose a router has built up the routing table shown below. The router can deliver packets directly over Interfaces 0 and 1, or forward packets to routers R2, R3, or R4. Describe what the router does with packets addressed to the given destinations (a) 128.96.39.10 (b) 128.96.40.12 (c) 128.96.40.151 (d) 192.4.153.17 (e) 192.4.153.90 [Table shown below]",
    "Q1_B": "Consider the following network with five routers A, B, C, D, and E using Link State Routing. (i) Show the link state packets (LSPs) prepared by each router. (ii) How node A builds the routing table using the forward search algorithm, showing each step clearly."
  }}
}}

---

Now extract this structured JSON from the following question paper text:

{text}
"""
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
