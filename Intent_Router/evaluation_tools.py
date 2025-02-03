from copy import deepcopy
import json
import ast
import os 
import json
import numpy as np
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch
# from peft import PeftModel
from vllm import LLM, SamplingParams

processor = AutoProcessor.from_pretrained("/root/sspaas-tmp/llm_for_ComSys/Qwen2-VL-7B-Instruct")
# model = Qwen2VLForConditionalGeneration.from_pretrained("/root/sspaas-tmp/Qwen2-VL-7B-Instruct", device_map="auto", torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2") 
# model = AutoModelForCausalLM.from_pretrained("/root/sspaas-tmp/llm_for_ComSys/DeepSeek-R1-Distill-Qwen-7B/", device_map="auto", torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2") 
model = LLM(model = "/root/sspaas-tmp/llm_for_ComSys/Qwen2-VL-7B-Instruct")
sampling_params = SamplingParams(temperature=0.5, top_p=0.8, repetition_penalty=1.05, max_tokens=8000)

syst_prompt = """
Please find out the intent of the user in monitoring Wireless Network Access. Intent include possibilities as follows:
 {
"access probability": ['fixed', 'varies'],
"efficiency": ['minimize', 'moderate', 'maximize']
}
The answer must be in JSON format as follows:
{
"access probability": {your choice},
"efficiency": {your choice}
}
"""

def check_questions_with_val_output(questions_dict, model, tokenizer):
    questions_only = deepcopy(questions_dict)
    answers_only = {}
    for q in questions_dict:
        answers_only[q] = {
            "question": questions_dict[q]["question"],
            "answer": questions_dict[q]["answer"]
        }
    
        questions_only[q].pop("answer")
        
        if 'explanation' in questions_only[q]:
            questions_only[q].pop('explanation')

        if 'category' in questions_only[q]:
            questions_only[q].pop('category')
    
    user_prompt = "Here are the questions: \n "
    user_prompt += json.dumps(questions_only)
    
    messages = [
        {
            "role": "system",
            "content": syst_prompt
        },
        {
            "role": "user",
            "content": user_prompt               
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer(text=[text], padding = True, return_tensors="pt").to("cuda")    
    generated_ids = model.generate(
        [text],
        sampling_params, # vllm使用
        # max_new_tokens = 8000, # transformers使用
        # do_sample = True,
        # temperature = 0.7, 

    )

    # transformer库时使用
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    # generated_output = tokenizer.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # predicted_answers_str = generated_output[0]
    
    predicted_answers_str = generated_ids[0].outputs[0].text
    check_answer = deepcopy(predicted_answers_str)

    # 检查{}是否匹配
    open_braces = predicted_answers_str.count('{')  
    close_braces = predicted_answers_str.count('}')  
    
    # 如果不匹配，在最后补全  
    if open_braces > close_braces:  
        predicted_answers_str += '\n}' * 1

    predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
    predicted_answers_str = predicted_answers_str.replace('\n```', '')  # DeepSeek R1模型专用
    predicted_answers_str = predicted_answers_str[predicted_answers_str.find("{"):] # 左侧截断
    predicted_answers_str = predicted_answers_str[:predicted_answers_str.rfind('}')+1]  # 右侧截断，针对CoT模型

    parsed_predicted_answers = ast.literal_eval(predicted_answers_str)
    
    for q in parsed_predicted_answers:
        if "answer" in parsed_predicted_answers[q] and "question" in parsed_predicted_answers[q]:
            parsed_predicted_answers[q] = {
                "question": parsed_predicted_answers[q]["question"],
                "answer": parsed_predicted_answers[q]["answer"]
            }
    
    accepted_questions = {}
    
    for q in questions_dict:
        if q in parsed_predicted_answers and q in answers_only:
            if parsed_predicted_answers[q] == answers_only[q]:
                accepted_questions[q] = questions_dict[q]

    return accepted_questions, parsed_predicted_answers, check_answer

def TeleQnA(model_name: str, n_questions:int, end:int, model, tokenizer):
    questions_path = "/root/sspaas-tmp/llm_for_ComSys/TeleQnA.txt"
    save_path = os.path.join("/root/sspaas-tmp/llm_for_ComSys/"+model_name+"_answers.txt")

    print("Evaluating {}".format(model_name))

    with open(questions_path, encoding="utf-8") as f:
        loaded_json = f.read()
    all_questions = json.loads(loaded_json)

    shuffled_idx = np.arange(len(all_questions))

    # 从中断点继续
    # if os.path.exists(save_path):
    #     with open(save_path) as f:
    #         loaded_json = f.read()
    #     results = json.loads(loaded_json)
        
    #     start = len(results)
    #     categories = [ques['category'] for ques in results.values()]
    #     correct = [ques['correct'] for ques in results.values()]
    # else:
    results = {}
    start = 105
    categories = []
    correct = []
        

    print("Start at question: {}".format(start))

    # k = 0

    for start_id in range(start, end, n_questions):
        end_id = np.minimum(start_id + n_questions, len(all_questions)-1)
                
        q_names = ["question {}".format(shuffled_idx[k]) for k in range(start_id, end_id)]
        selected_questions = {}
        
        for q_name in q_names:
            selected_questions[q_name] = all_questions[q_name]

        accepted_questions, parsed_predicted_answers, check_answer = check_questions_with_val_output(selected_questions, model, tokenizer)
        
        for q in selected_questions:  
            answer = parsed_predicted_answers[q]['answer'] if 'answer' in parsed_predicted_answers[q] else parsed_predicted_answers['answer'] # 提高答案检测鲁棒性
            results[q] = deepcopy(selected_questions[q])
            results[q]['tested answer'] = answer
            results[q]['correct'] = q in accepted_questions
            correct += [results[q]['correct']]
            categories += [selected_questions[q]['category']]
        
        # 定时汇报
        # k += 1
        # if k % 3 == 0:
        #     with open(save_path, 'w') as f:
        #         res_str = json.dumps(results)
        #         f.write(res_str)

        # 生成小结-
        res = pd.DataFrame.from_dict({
            'categories': categories,
            'correct': correct
        })

        summary = res.groupby('categories').mean()
        summary['counts'] = res.groupby('categories').count()['correct'].values
        
        print("Total number of questions answered: {}".format(len(categories)))
        print(summary)

    # 保存回答
    # with open(save_path, 'w') as f:
    #     res_str = json.dumps(results)
    #     f.write(res_str)

    res = pd.DataFrame.from_dict({
        'categories': categories,
        'correct': correct
    })

    # summary = res.groupby('categories').mean()
    # summary['counts'] = res.groupby('categories').count()['correct'].values

    summary = res.groupby('categories').agg({  
        'correct': 'mean',  
        'categories': 'count'  
    }).rename(columns={'categories': 'counts'})

    print("Total number of questions answered: {}".format(len(categories)))
    print(summary)

    return summary


if __name__ == '__main__':
    # 被测适配器
    adapter = ['2048']
    ckpoint = [ '50', '100', '111']
    # 参数
    results_path = "/root/sspaas-tmp/llm_for_ComSys/Qwen2-VL.csv"
    step = 1
    question_num = 120

    dataframes = {}
    # 先测试原版
    dataframes['original'] = TeleQnA('original', step, question_num, model, processor)

    # 再测试微调版
    for i in range(len(adapter)):
        for j in range(len(ckpoint)):
            adapter_path = "/root/sspaas-tmp/llm_for_ComSys/qwen_qlora/Qwem2.5-14B-test/checkpoint-"+ckpoint[j]
            adapter_name = adapter[i]+"+"+ckpoint[j]
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            model.set_adapter(adapter_name)  
            dataframes[adapter_name] = TeleQnA(adapter_name, step, question_num, model, processor)

    # 创建一个新 DataFrame 用于存储汇总结果  
    results = pd.DataFrame()  

    # 循环添加每个汇总的正确率到结果 DataFrame  
    for summary_name, df in dataframes.items(): 
        # 获取每个 summary 的正确率并添加到结果中  
        correct_rates = df['correct']  

        # 将结果合并到新的结果 DataFrame  
        if results.empty:  
            results[summary_name] = correct_rates  # 第一次赋值  
        else:  
            results[summary_name] = correct_rates  # 后续追加  

    # 转换列名为 categories 名称  
    results.insert(0, 'categories', correct_rates.index)  # 添加 categories 列  

    # 如果需要重新排列列顺序，将 categories 列挪到最左边  
    final_column_order = ['categories'] + list(dataframes.keys())  
    results = results[final_column_order]  

    # 保存结果为 CSV 文件  
    results.to_csv(results_path, index=False) 


