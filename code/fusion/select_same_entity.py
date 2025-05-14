# Please install dependencies first: 
# pip3 install openai pandas

import csv
import os
from openai import OpenAI
import json
import datetime
import re
import traceback

import pandas as pd
from tqdm import tqdm


def process_medical_terms(config,csv_path,output_file, batch_size=20):
    # 读取CSV文件
    # 先通过open解决部分行有字节编码错误，errors='ignore'
    with open(csv_path, encoding='utf-8',errors='ignore') as f:
        df = pd.read_csv(f,sep=',')

    term_pairs = df[['Name1', 'Name2']].values.tolist()
     # 调用API
    client = OpenAI(
        api_key=config.get("GPT_Api","api_key"),
        base_url=config.get("GPT_Api","base_url"),
    )

    all_results = []
    total = len(term_pairs)
    for start_idx in tqdm(range(5160, total, batch_size)):
        try:
            end_idx = start_idx + batch_size
            current_batch = term_pairs[start_idx:end_idx]
        
            # 构建查询内容
            query = "判断以下术语对是否为同一临床概念：\n"
            for pair in current_batch:
                query += f"{pair[0]},{pair[1]}\n"

            response = client.chat.completions.create(
                model=config.get("GPT_Api","llm_model"),
                messages=[
                    {"role": "system", "content": "医学专家#任务：临床概念匹配#规则：1.仅输出紧凑JSON 2.Is_same必须为布尔值"},
                    {"role": "user", "content": f"{query}\n生成包含Entity1、Entity2、Is_same的JSON数组，格式示例：[{{\"Entity1\":\"Nedaplatin\", \"Entity2\":\"CDGP\", \"Is_same\":true}}]"}
                ],
                response_format={"type": "json_object"},  # 强制JSON模式
                temperature=0.5,
                stream=False
            )
            jsonStr = response.choices[0].message.content
            
            result = json.loads(jsonStr,strict=False)
            all_results.extend(result)
            print(result)
            # if 'results' in result.keys():
            #     result = result['results']
            #     # 添加原始术语对匹配
            #     for i, pair in enumerate(current_batch):
            #         if i < len(result):
            #             result[i]["term1_original"] = pair[0]
            #             result[i]["term2_original"] = pair[1]
            # else:
            #     result["term1_original"] = pair[0]
            #     result["term2_original"] = pair[1]
        except Exception as e:
            print(f"Process interrupted: {str(e)}")
            print(f"错误位置：第 {e.lineno} 行，第 {e.colno} 列")
            print("上下文：", jsonStr[e.pos-50:e.pos+50])  # 显示错误附近的上下文
            traceback.print_exc()
            continue
    save_results(all_results, output_file)
    
    return all_results
           
def save_results(data, filename):
    try:
        current_time = datetime.datetime.now().strftime(r'%Y%m%d%H%M%S')
        with open(f"{filename}_{current_time}.csv", 'w', encoding='utf-8') as f:
            # 创建CSV写入器
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            # 写入表头
            writer.writeheader()
            # 写入数据行
            writer.writerows(data)
        print(f"Results saved to {filename}")
    except Exception as e:
        current_time = datetime.datetime.now().strftime(r'%Y%m%d%H%M%S')
        file_path = os.path.dirname(filename)
        print(f"Failed to save results: {str(e)}")
        with open(f"{file_path}/emergency_backup_{current_time}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)



# 使用示例
if __name__ == "__main__":
    csv_file = "../../raw_data/知识图谱_v0318/全量级知识图谱_v0318_Eng/实体相似度/疾病实体表_v1.8/Special_High.csv"  
    output_file = "../../raw_data/知识图谱_v0318/全量级知识图谱_v0318_Eng/实体相似度/疾病实体表_v1.8/sh_results_selected"
    query_batch = 50 #每次设置50为一组进行查询
    results = process_medical_terms(csv_file,output_file,30)
    
    # 打印结果
    print(json.dumps(results, indent=2, ensure_ascii=False))
