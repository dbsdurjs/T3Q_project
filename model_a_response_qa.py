#--------------------------------------------------------------------------------------------------------------------
# model name : meta-llama/Llama-3.1-8B
#--------------------------------------------------------------------------------------------------------------------
import transformers, torch
import csv, os
from utils import read_prompts_from_csv, read_prompts_from_txt

device_a = 0 if torch.cuda.is_available() else -1

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
_pipeline_a = None

def get_pipeline_a():
    global _pipeline_a
    if _pipeline_a is None:
        _pipeline_a = transformers.pipeline(
            "text-generation",
            model=model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=device_a,
        )
        _pipeline_a.model.eval()
    return _pipeline_a

def get_llm_a_response(prompt):
    pipeline = get_pipeline_a()
    instruction = '이 질문에 대한 답변을 4~5문장 이내로 한국어로 완전하게 작성해줘.'
    
    # 🔹 입력 타입 처리
    single_input = False
    if isinstance(prompt, str):
        prompts = [prompt]
        single_input = True
    elif isinstance(prompt, list):
        prompts = prompt
    else:
        raise ValueError("prompt must be string or list of strings")

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    responses = []
    for p in prompts:
        messages = [
            {"role" : "system", "content" : f"{p}"},
            {"role" : "user", "content" : f"{instruction}"}
        ]
        
        formatted = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = pipeline(
            formatted,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        
        response = outputs[0]['generated_text'][len(formatted):]
        responses.append(response.strip())
    
    # 🔹 return 타입 구분
    if single_input:
        return responses[0], model_id
    else:
        return responses, model_id



if __name__ == "__main__":
    file_path = "./defense_questions_snunlp.txt"
    output_path = "./defense_response_model_a_snunlp.txt"

    if file_path.endswith('.csv'):
        prompts = read_prompts_from_csv(file_path)
    elif file_path.endswith('.txt'):
        prompts = read_prompts_from_txt(file_path)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for prompt in prompts:
            answer = get_llm_a_response(prompt)
            print(f"Prompt: {prompt}\nResponse: {answer}\n")
            out_f.write(f"{{prompt:{prompt}, response_a:{answer}}}\n")