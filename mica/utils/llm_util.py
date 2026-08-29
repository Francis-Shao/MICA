from pathlib import Path

import openai
import yaml
import time

current_dir = Path(__file__).parent
config_path = current_dir.parent.parent / "configs" / "configs.yaml"

print(config_path)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

client = openai.OpenAI(
    base_url=config["base_url"],
    api_key=config["api_key"]
)

def handler(signum, frame):
    raise Exception("End of time")


def request_response(content):
    res = None
    while res is None:
        try:
            res = client.chat.completions.create(
                model=config["llm_model"],
                messages=content,
                temperature=config["temperature"]
            )
        except openai._exceptions.BadRequestError as e:
            print(e)
        except openai._exceptions.RateLimitError as e:
            print("Rate limit exceeded. Waiting...")
            print(e)
            time.sleep(5)
        except openai._exceptions.APIConnectionError as e:
            print("API connection error. Waiting...")
            time.sleep(5)
        except Exception as e:
            print(e)
            print("Unknown error. Waiting...")
            time.sleep(1)
    return res

if __name__ == "__main__":
    response_text = request_response([{"role": "user", "content": "hello"}])
    print(response_text)