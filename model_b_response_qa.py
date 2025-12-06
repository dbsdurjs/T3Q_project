#--------------------------------------------------------------------------------------------------------------------
# model name : Qwen/Qwen3-8B
#--------------------------------------------------------------------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import read_prompts_from_csv, read_prompts_from_txt

# 물리 GPU 1번 사용 (없으면 CPU)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    device_b = torch.device("cuda:1")
elif torch.cuda.is_available():
    device_b = torch.device("cuda:0")
else:
    device_b = torch.device("cpu")

# Load model and tokenizer
model_id = "LiquidAI/LFM2-2.6B"
_model_b = None
_tokenizer_b = None

def get_model_b():
    global _model_b, _tokenizer_b
    if _model_b is None:
        _tokenizer_b = AutoTokenizer.from_pretrained(model_id)
        _model_b = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if device_b.type == "cuda" else torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        _model_b.to(device_b)
        _model_b.eval()
    return _model_b, _tokenizer_b

def get_llm_b_response(prompt):
    model, tokenizer = get_model_b()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Generate answer
    full_prompt = f"{prompt}. 이 질문에 대한 답변을 4~5문장 이내로 한국어로 완전하게 작성해줘."

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )
    generated_ids = output[0][input_ids.shape[-1]:]  # 입력 길이 이후만 추출

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), model_id

if __name__ == "__main__":
    file_path = "./defense_questions_snunlp.txt"
    output_path = "./defense_response_model_b_snunlp.txt"

    if file_path.endswith('.csv'):
        prompts = read_prompts_from_csv(file_path)
    elif file_path.endswith('.txt'):
        prompts = read_prompts_from_txt(file_path)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for prompt in prompts:
            answer = get_llm_b_response(prompt)
            print(f"Prompt: {prompt}\nResponse: {answer}\n")
            out_f.write(f"{{prompt:{prompt}, response_b:{answer}}}\n")