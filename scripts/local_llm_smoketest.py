import argparse
import os
import sys

import requests

try:
    from openai import OpenAI
except ImportError:  # DeepSeek call is optional
    OpenAI = None


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_LOCAL_BASE = "http://localhost:11434"
LOCAL_LLM_BASE = os.getenv("LOCAL_LLM_BASE", DEFAULT_LOCAL_BASE)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "deepseek-r1:14b")
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "180"))

DEFAULT_PROMPT = (
    "Please help me plan a trip from St. Petersburg to Rockford spanning 3 days "
    "from March 16th to March 18th, 2022. The travel should be planned for a "
    "single person with a budget of $1,700."
)


def DeepSeek_response(messages: str, model_name: str = "deepseek-chat") -> str:
    """Call the DeepSeek API with an OpenAI-compatible client."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    if OpenAI is None:
        raise RuntimeError("pip install openai>=1.0.0 is required for DeepSeek API")

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1",
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def Local_response(messages: str, model_name: str = "local", base_url: str = None,
                   timeout: float = None) -> str:
    """Send a prompt to a locally hosted LLM endpoint."""
    base = base_url or LOCAL_LLM_BASE or DEFAULT_LOCAL_BASE

    chosen_model = model_name if model_name not in ("local", None) else LOCAL_LLM_MODEL
    payload = {
        "model": chosen_model,
        "prompt": messages,
        "temperature": 0.0,
        "stream": False,  # Ollama-compatible non-streaming response
    }

    endpoint = f"{base.rstrip('/')}/api/generate"
    resp = requests.post(endpoint, json=payload, timeout=timeout or LOCAL_LLM_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"Local LLM request failed {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except ValueError:
        raise ValueError(
            f"Local LLM returned non-JSON (status={resp.status_code}, len={len(resp.text)}): {resp.text[:200]}"
        )

    for key in ("response", "text", "output", "content"):
        if key in data:
            return data[key]

    raise KeyError("Local LLM response JSON did not include a supported text field.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal smoke test to query a local LLM for the travel plan prompt."
    )
    parser.add_argument("--backend", choices=["local", "deepseek"], default="local",
                        help="Choose which client to call.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to the model.")
    parser.add_argument("--model", default=None, help="Model name/path override.")
    parser.add_argument("--base", default=None, help="Local LLM base URL (overrides LOCAL_LLM_BASE).")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds (default 180).")
    args = parser.parse_args()

    try:
        if args.backend == "local":
            reply = Local_response(args.prompt, model_name=args.model, base_url=args.base, timeout=args.timeout)
        else:
            reply = DeepSeek_response(args.prompt, model_name=args.model or "deepseek-chat")
    except Exception as exc:  # pragma: no cover - simple CLI helper
        sys.stderr.write(f"Request failed: {exc}\n")
        sys.exit(1)

    print("\n=== Model response ===\n")
    print(reply)


if __name__ == "__main__":
    main()
