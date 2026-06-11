from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
 
model_path = "./tinyllama-coding-model"
 
print("Loading model... Please wait.")
 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
 
print("Coding Assistant Ready!")
print("Type 'exit' to quit.\n")
 
while True:
    prompt = input("You: ")
 
    if prompt.lower() == "exit":
        print("Goodbye!")
        break
 
    text = f"""
You are a professional coding assistant.
Give correct, beginner-friendly, clean programming answers.
If code is requested, provide complete working code.
 
Instruction: {prompt}
 
Answer:
"""
 
    inputs = tokenizer(text, return_tensors="pt")
 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
 
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
    answer = result.replace(text, "").strip()
 
    print("\nAI:", answer)
    print()
 