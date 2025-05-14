#time:2025/03/07
#function:根据概念变更情况，对关系表进行更新
from collections import defaultdict
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..'))
from util.FileUtil import FileUtil

import ast
from pathlib import Path



def update_relations(change_file,change_type, data_dir,out_put_dir):
    # 读取概念变更记录，只保留MergeConcepts类型
    changes = pd.read_csv(change_file)
    # merge_concepts = changes[changes['type'] == 'MergeConcepts']
    merge_concepts = changes
    
    # 构建概念映射字典：旧概念 -> 新概念
    concept_mapping = {}
    for _, row in merge_concepts.iterrows():
        from_concepts = row['from_concept']
        to_concept = row['to_concept']
        concept_mapping[from_concepts] = to_concept

    # 遍历目录下所有包含"症状"的CSV文件
    symptom_files = [f for f in Path(data_dir).glob('*.xlsx') if change_type in f.stem]
    print(symptom_files)

    

    for file in symptom_files:
        print(f"处理文件: {file.name}")
        
        # 读取数据
        # --- 修改点：优先从输出目录读取文件 ---
        output_file = Path(out_put_dir) / file.name
        if output_file.exists():
            # 如果输出目录已有该文件，则从输出目录读取
            df = pd.read_excel(output_file)
        else:
            # 否则从原始目录读取
            df = pd.read_excel(file)
        
        # # 更新Subject列
        # df['Subject'] = df['Subject'].apply(
        #     lambda x: concept_mapping.get(x, x))
        
        # # 更新Object列（如果包含概念ID）
        # if 'Object' in df.columns:
        #     df['Object'] = df['Object'].apply(
        #         lambda x: concept_mapping.get(x, x))

        # 主动遍历字典的键，替换数据中的匹配项
        for old_concept, new_concept in concept_mapping.items():
            # 替换 Subject 列
            df.loc[df['Subject'] == old_concept, 'Subject'] = new_concept
            # 替换 Object 列（如果存在）
            if 'Object' in df.columns:
                df.loc[df['Object'] == old_concept, 'Object'] = new_concept

        
        df = df.drop_duplicates(subset=['Subject', 'Object'], keep='first');
        FileUtil.ensure_directory_exists(out_put_dir + file.name)
        # 保存更新后的文件
        print("output_dir:" + out_put_dir)
        print(out_put_dir + file.name)
        df.to_excel(out_put_dir + "/" + file.name, index=False)
        print(f"已更新并备份: {file.name}")

def analyze_conceptid_discrepancy(root_path, relation_files, entity_files, print_report=True):
    """
    基于ConceptId分析关系表与实体表的差集及未使用的实体类型
    
    参数：
    relation_files (list): 关系表文件路径列表（需包含Subject和Object列）
    entity_files (list): 实体表文件路径列表（需包含ConceptId列）
    print_report (bool): 是否打印分析报告（默认True）

    返回值：
    tuple: (差集字典, 未使用实体类型集合)
      - 差集字典格式: {实体类型: 缺失的ConceptId集合}
      - 未使用实体类型集合: set()
    """
    # 读取关系表中的ConceptId并分类
    relation_concepts = defaultdict(set)
    for r_file in relation_files:
        try:
            df = pd.read_excel(f"{root_path}/Relations/{r_file}", engine="openpyxl")
            print(f"关系表 {r_file} 读取成功")
            # 提取Subject和Object列的ConceptId
            subjects = df["Subject"].dropna().tolist()
            objects = df["Object"].dropna().tolist()
            # 按前缀分类（如DRUG_C0001 -> DRUG）
            for cid in subjects + objects:
                if "_" in cid:
                    entity_type = cid.split("_")[0]
                    relation_concepts[entity_type].add(cid)
        except Exception as e:
            print(f"关系表 {r_file} 读取失败: {str(e)}")

    # 读取实体表中的ConceptId并分类
    entity_concepts = defaultdict(set)
    for e_file in entity_files:
        try:
            df = pd.read_excel(f"{root_path}/Entities/{e_file}", engine="openpyxl")
            cids = df["ConceptId"].dropna().tolist()
            for cid in cids:
                if "_" in cid:
                    entity_type = cid.split("_")[0]
                    entity_concepts[entity_type].add(cid)
        except Exception as e:
            print(f"实体表 {e_file} 读取失败: {str(e)}")

    # 计算差集
    discrepancy = defaultdict(set)
    for entity_type, cids in relation_concepts.items():
        missing = cids - entity_concepts.get(entity_type, set())
        if missing:
            discrepancy[entity_type] = missing

    # 检查未使用的实体类型
    unused_types = set(entity_concepts.keys()) - set(relation_concepts.keys())

    # 打印报告
    if print_report:
        print("="*40 + "\nConceptID差集分析报告\n" + "="*40)
        for entity_type, missing in discrepancy.items():
            sample = list(missing)[:10]
            print(f"\n❌ 类型 [{entity_type}] 缺失 {len(missing)} 个ConceptID，例如：")
            print(", ".join(sample) + ("..." if len(missing)>5 else ""))
        
        print("\n" + "="*30 + "\n未使用的实体类型\n" + "="*30)
        for t in unused_types:
            print(f"⚠️ 类型 [{t}] 在实体表中定义但未被任何关系表引用")

    return dict(discrepancy), unused_types


if __name__ == "__main__":
    # 参数配置
    change_file = "../../raw_data/知识图谱_v0318/全量级知识图谱_v0318_Eng/实体相似度/疾病实体表_v1.8/concept_changes.csv"
    change_type = "疾病"
    data_dir = "../../raw_data/知识图谱_v0318/全量级知识图谱_v0318_Eng/Relations"
    out_put_dir = "../../raw_data/知识图谱_v0318/全量级知识图谱_v0318_update/Relations/"
    # 执行更新
    update_relations(change_file, change_type, data_dir,out_put_dir)
    print(f"所有{change_type}相关表格更新完成！")



    #输入文件列表
    root_path = "../../raw_data/知识图谱_v0318/全量级知识图谱_v0318_update/"
    relation_files = [
       "基因_疾病_v2.1.xlsx", "基因_通路_v2.0.xlsx", "成分_通路_v2.0.xlsx",
        "症状_基因_v2.0.xlsx",  "症状_疾病_v2.1.xlsx",
        "西药_基因_v2.3.xlsx", "西药_症状_v2.1.xlsx", "西药_通路_v2.1.xlsx"
    ]
    entity_files = [
        "中药实体表_v1.5.xlsx", "基因实体表_v1.5.xlsx", "成分实体表_v1.0.xlsx",
        "疾病实体表_v1.8.xlsx", "病因实体表_v1.0.xlsx", "症状实体表_v1.1.xlsx",
        "西药实体表_v1.6.xlsx", "通路实体表_v1.1.xlsx"
    ]

    # 调用方法
    diff, unused = analyze_conceptid_discrepancy(root_path, relation_files, entity_files)