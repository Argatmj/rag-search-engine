import argparse
import mimetypes
import base64
from lib.llm import Model, image_prompt

def main():
    parser = argparse.ArgumentParser(description="Describe an image")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--query", type=str, help="The query")

    args = parser.parse_args()
    llm = Model()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as img:
        img_content = img.read()

    data_url = f"data:{mime};base64,{base64.b64encode(img_content).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": image_prompt().strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]
    llm.get_response_multimodal(messages)
    
if __name__ == "__main__":
    main()