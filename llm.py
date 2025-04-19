from PIL import Image
from io import BytesIO
import base64
import ollama
from suggestion_state import set_suggestion

from pydantic import BaseModel
from typing import List


class SuggestionResponse(BaseModel):
    message: str
    recommendation: List[str]


def generate_suggestion(image_array):
    try:
        # Convert image to base64
        # Convert BGR to RGB
        rgb_image = image_array[:, :, ::-1]
        pil_img = Image.fromarray(rgb_image)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Define prompt
        prompt = '''
Look at the person's appearance and suggest a fun snack combo for them, using this menu:

☕ Coffee  
🍿 Popcorn (Butter or Kettlecorn)  
🧃 Juice (Apple, Grape, or Cranberry)

Your response must be a short, quirky, message and related to the person visible in the image in the `message` field, and a `recommendation` array with at most one popcorn (Butter Popcorn or Kettlecorn Popcorn), and at most one drink (Coffee or a Juice). If nothing fits, leave the array empty. The message should include something about the person like the color of their hair, or if they are wearing glasses, or the color of their clothes, etc. The recommendation should be a list of strings with the exact names of the items from the menu. For example:
Return only the JSON object, no extra text.
'''

        response = ollama.chat(
            model='gemma3:12b',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_base64]
            }],
            format=SuggestionResponse.model_json_schema(),
        )

        # Parse LLM response
        suggestion = SuggestionResponse.model_validate_json(
            response['message']['content'])
        print("[OLLAMA RESPONSE]", suggestion)

        # Save structured object to state
        set_suggestion(suggestion.model_dump(), img_base64)

    except Exception as e:
        print("[ERROR in LLM call]", e)
