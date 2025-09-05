"""
pip install transformers --upgrade
pip install openai-harmony

# this is run with transformers
python3 gpt-oss.py

very slow
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "openai/gpt-oss-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    messages = [
        {"role": "user", "content": "Explain what MXFP4 quantization is."},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7)
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
