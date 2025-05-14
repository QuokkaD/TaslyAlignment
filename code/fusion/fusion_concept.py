import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

class ConceptFusion:
    def __init__(self, root_path, entities_path, output_file):
        # 配置参数
        self.root_path = root_path
        self.entities_path = entities_path
        self.output_file = output_file
        
        # 数据存储
        self.entities = None
        self.sh_results = None
        self.sl_results = None
        self.concept_entities = defaultdict(set)
        self.entity_concept = {}
        self.new_cid_counter = 0
        self.change_log = []
        self.processed_pairs = set()

    def load_data(self):
        """加载所有数据源"""
        # 加载实体表
        self.entities = pd.read_excel(self.entities_path)
        self.entities['AtomId'] = self.entities['AtomId'].astype(str)
        
        # 加载相似度结果
        self._load_similarity_data('sh')
        #print(self.sh_results)
        self._load_similarity_data('sl')
        #print(self.sl_results)
        
        # 初始化概念结构
        self._init_concept_structure()

    

    def _load_similarity_data(self, prefix):
        """加载相似度文件"""
        files = glob.glob(f"{self.root_path}/{prefix}_results_selected*.csv")
        print("加载相似度文件：" + files[0])
        df = pd.read_csv(files[0])
        df = self._add_entity_ids(df)
        setattr(self, f"{prefix}_results", df.dropna())

    def _add_entity_ids(self, df):
        """添加实体ID列"""
        name_to_id = dict(zip(self.entities["Name"], self.entities["AtomId"]))
        df["Entity1_ID"] = df["Entity1"].map(name_to_id)
        df["Entity2_ID"] = df["Entity2"].map(name_to_id)
        return df

    def _init_concept_structure(self):
        """初始化概念数据结构"""
        for _, row in self.entities.iterrows():
            self.concept_entities[row["ConceptId"]].add(row["AtomId"])
            self.entity_concept[row["AtomId"]] = row["ConceptId"]
        
        # 初始化新概念ID计数器
        max_cid = self.entities["ConceptId"].str.extract(r'C(\d+)')[0].max()
        self.new_cid_counter = int(max_cid) + 1

    def _process_high_similarity(self):
        """处理高相似度实体对"""
        for _, row in tqdm(self.sh_results.iterrows(), total=len(self.sh_results)):
            if row["Is_same"] in [True, 'True', 'Yes', 'yes', 'same']:
                e1, e2 = str(row["Entity1_ID"]), str(row["Entity2_ID"])
                pair = frozenset([e1, e2])
                
                if pair in self.processed_pairs:
                    continue
                self.processed_pairs.add(pair)
                
                self._merge_entities(e1, e2)

    def _merge_entities(self, e1, e2):
        """合并两个实体"""
        c1 = self.entity_concept.get(e1)
        c2 = self.entity_concept.get(e2)
        if c1 == c2:
            return
        # 处理单实体概念
        if self._handle_single_entity(e1, c2) or self._handle_single_entity(e2, c1):
            return
        
        # 处理多实体概念合并
        self._check_and_merge(e1,e2,c1,c2)

    def _handle_single_entity(self, entity, target_cid):
        """处理单实体概念的合并"""
        if len(self.concept_entities[self.entity_concept[entity]]) == 1:
            origin_cid = self.entity_concept[entity]
            if origin_cid != target_cid:
                # 迁移实体到目标概念
                self.concept_entities[target_cid].add(entity)
                self.concept_entities[origin_cid].remove(entity)
                del self.concept_entities[origin_cid]
                self.entity_concept[entity] = target_cid
                
                # 记录变更
                self.change_log.append({
                    "type": "MergeSingle",
                    "from_concept": origin_cid,
                    "to_concept": target_cid,
                    "entities": [entity]
                })
            return True
        return False
  

    def _merge_concepts(self, c1, c2):
        """合并两个概念"""
        # 冲突检查（简化示例）
        print(self.sl_results)
        conflict = self.sl_results[
            ((self.sl_results["Entity1_ID"].isin(self.concept_entities[c1])) & 
             (self.sl_results["Is_same"] == True)) |
            ((self.sl_results["Entity2_ID"].isin(self.concept_entities[c1])) & 
             (self.sl_results["Is_same"] == True))
        ].any()
        
        if not conflict.any():
            # 合并概念
            self.concept_entities[c1].update(self.concept_entities[c2])
            # 删除概念 ID 前检查是否存在
            if c2 in self.concept_entities:
                del self.concept_entities[c2]
            for e in self.concept_entities[c1]:
                self.entity_concept[e] = c1
            self.change_log.append({
                "type": "MergeConcepts",
                "from_concept": c2,
                "to_concept": c1,
                "entities": list(self.concept_entities[c1])
            })
    def _check_and_merge(self,e1,e2,c1, c2):
            global new_cid_counter
            # 检查c1中的实体是否与c2的实体相似
            should_merge = True
            for e in self.concept_entities[c1]:
                # 检查是否在sl_result中存在冲突

                conflict = self.sl_results[
                    ((self.sl_results["Entity1_ID"] == e) & (self.sl_results["Is_same"] == True)) |
                    ((self.sl_results["Entity2_ID"] == e) & (self.sl_results["Is_same"] == True))
                ].any()
                if conflict.any():
                    should_merge = False
                    break
            if should_merge:
                # 创建新概念或合并到现有概念
                self.concept_entities[c1].update(self.concept_entities[c2])
                # 删除概念 ID 前检查是否存在
                if c2 in self.concept_entities:
                    del self.concept_entities[c2]
                # 更新实体映射
                for e in self.concept_entities[c1]:
                    self.entity_concept[e] = c1
                # 记录变更
                self.change_log.append({
                    "type": "MergeConcepts",
                    "from_concept": c2,
                    "to_concept": c1,
                    "entities": self.concept_entities[c1]
                })
            else:
                # 只将当前实体合并到目标概念
                self.concept_entities[c2].add(e1)
                self.concept_entities[c1].remove(e1)
                self.entity_concept[e1] = c2
                # 记录变更
                self.change_log.append({
                    "type": "MergeSingle",
                    "from_concept": c1,
                    "to_concept": c2,
                    "entities": [e1]
                })

    def _save_results(self):
        """保存处理结果"""
        # 保存变更记录
        change_path = f"{self.root_path}/concept_changes.csv"
        pd.DataFrame(self.change_log).to_csv(change_path, index=False)
        
        # 生成最终映射表
        final_mapping = pd.DataFrame([
            {"AtomId": e, "NewConceptId": c}
            for c, entities in self.concept_entities.items()
            for e in entities
        ])
        final_mapping.to_csv(f"{self.root_path}/final_entity_concepts.csv", index=False)
        return change_path,final_mapping

    def update_entities(self,final_mapping):
        # 读取数据
        original_df = self.entities
        new_concept_df = final_mapping

        # 删除旧ConceptId列
        original_df.drop('ConceptId', axis=1, inplace=True)

        # 合并新ConceptId
        merged_df = pd.merge(original_df, new_concept_df, on='AtomId', how='left')

        # 重命名并调整列顺序
        merged_df.rename(columns={'NewConceptId': 'ConceptId'}, inplace=True)
        merged_df = merged_df[['AtomId', 'ConceptId', 'Name', 'Source']]

        # 保存结果
        merged_df.to_excel(f"{self.output_file}", index=False)

    def process_fusion(self):
        """执行概念融合主流程"""
        # 处理高相似度结果
        self._process_high_similarity()
        
        # 处理低相似度结果（示例保留原逻辑）
        # self._process_low_similarity()
        # 生成结果文件
        change_path, final_mapping = self._save_results()

        self.update_entities(final_mapping)

        return change_path

