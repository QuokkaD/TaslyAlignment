# time:2025/03/10
# function:根据大模型结果对实体进行拆分
import copy
import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

class ConceptSeparator:
    def __init__(self, root_path, entities_path):
        # 配置参数
        self.root_path = root_path
        self.entities_path = entities_path
        self.output_file = f"{root_path}/entity_sep_update.csv"
        
        # 数据存储
        self.entities = pd.read_excel(entities_path)
        self.sl_results = self._load_similarity_data()
        self._init_concept_data()  # 合并数据初始化逻辑
        
        # 状态管理
        self.new_cid_counter = self._get_max_cid() + 1
        self.change_log = []

    def _load_similarity_data(self):
        """合并加载相似度数据和ID映射"""
        files = glob.glob(f"{self.root_path}/sl_results_selected*.csv")
        df = pd.read_csv(files[0])
        
        # 合并实体ID映射
        name_to_id = dict(zip(self.entities["Name"], self.entities["AtomId"]))
        return df.assign(
            Entity1_ID = lambda x: x["Entity1"].map(name_to_id),
            Entity2_ID = lambda x: x["Entity2"].map(name_to_id)
        ).dropna()

    def _init_concept_data(self):
        """合并初始化概念数据结构"""
        self.concept_entities = defaultdict(set)
        self.entity_concept = {}
        
        for _, row in self.entities.iterrows():
            self.concept_entities[row["ConceptId"]].add(row["AtomId"])
            self.entity_concept[row["AtomId"]] = row["ConceptId"]

    def _get_max_cid(self):
        """获取最大概念ID"""
        return int(self.entities["ConceptId"].str.extract(r'C(\d+)')[0].max())

    def process_separation(self):
        """主处理流程"""
        # 处理低相似度实体对
        target_pairs = self.sl_results[
            self.sl_results["Is_same"].astype(str).str.lower().isin(["false", "no"])
        ]
        
        for _, row in tqdm(target_pairs.iterrows(), desc="Splitting concepts"):
            e1, e2 = str(row["Entity1_ID"]), str(row["Entity2_ID"])
            if self.entity_concept[e1] == self.entity_concept[e2]:
                self._split_single_concept(self.entity_concept[e1])

        # 保存结果
        result_df = self._save_outputs()
        return self.output_file,result_df

    def _split_single_concept(self, old_cid):
        """执行单个概念的拆分"""
        for entity in copy.deepcopy(self.concept_entities[old_cid]):
            new_cid = f"Disease_C{str(self.new_cid_counter).zfill(6)}"
            
            # 迁移实体
            self.concept_entities[old_cid].remove(entity)
            self.concept_entities[new_cid].add(entity)
            self.entity_concept[entity] = new_cid
            
            # 记录变更
            self.change_log.append({
                "type": "SplitConcept",
                "from_concept": old_cid,
                "to_concept": new_cid,
                "entities": [entity]
            })
            self.new_cid_counter += 1
        
        # 清理空概念
        if not self.concept_entities[old_cid]:
            del self.concept_entities[old_cid]

    def _save_outputs(self):
        """合并保存所有输出"""
        # 生成映射表
        final_mapping = pd.DataFrame([
            {"AtomId": e, "NewConceptId": c}
            for c, entities in self.concept_entities.items()
            for e in entities
        ])
        
        # 保存变更记录
        pd.DataFrame(self.change_log).to_csv(f"{self.root_path}/concept_splits_log.csv", index=False)
        
        # 更新实体表
        (self.entities.drop("ConceptId", axis=1)
          .merge(final_mapping, on="AtomId")
          .rename(columns={"NewConceptId": "ConceptId"})
        )
        self.entities.to_csv(self.output_file, index=False)
        return self.entities

# 使用示例
if __name__ == "__main__":
    processor = ConceptSeparator(
        root_path="../../raw_data/知识图谱_v0318/全量级知识图谱_v0318/实体相似度/疾病实体表_v1.8/",
        entities_path="../../raw_data/知识图谱_v0318/全量级知识图谱_v0318/Entities/疾病实体表_v1.8.xlsx",
    )
    print(processor.process_separation())