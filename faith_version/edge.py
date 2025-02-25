from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "THUDM/glm-edge-1.5b-chat"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

message = [{"role": "user", "content": "hello!"}]

inputs = tokenizer.apply_chat_template(
    message,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

generate_kwargs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "max_new_tokens": 128,
    "do_sample": False,
}
out = model.generate(**generate_kwargs)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
