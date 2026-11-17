import argparse
import os
from pathlib import Path

from openai import OpenAI


def create_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")
    return OpenAI()


def upload_file(client: OpenAI, file_path: Path) -> str:
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    with file_path.open("rb") as file:
        uploaded = client.files.create(
            file=file,
            purpose="user_data",
        )
    return uploaded.id


def summarize_file(client: OpenAI, file_id: str, model: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "请总结这篇文件的核心内容、方法、假设和结论。"},
                    {
                        "type": "input_file",
                        "file_id": file_id,
                    },
                ],
            }
        ],
    )
    return resp.output_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a file and summarize it with the OpenAI Responses API.")
    parser.add_argument("file", type=Path, help="Path to the file to summarize.")
    parser.add_argument("--model", default="gpt-5.5", help="OpenAI model ID. Defaults to gpt-5.5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = create_client()
    file_id = upload_file(client, args.file)
    print("Success, file_id =", file_id)

    result = summarize_file(client, file_id, args.model)
    print("\nOutput:\n")
    print(result)


if __name__ == "__main__":
    main()
