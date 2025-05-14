# main.py
import sys
import os
import configparser
from pathlib import Path

from fusion.divorce_concept import ConceptSeparator

# 添加Util目录到系统路径
sys.path.append(os.path.abspath('..'))

from util.FileUtil import FileUtil
from fusion.filterfile import DataFilter
from fusion.cal_similarity_paraller import OptimizedProcessor
from fusion.select_same_entity import process_medical_terms
from fusion.fusion_concept import ConceptFusion
from fusion.update_relation import update_relations

class PipelineRunner:
    def __init__(self, config_file="config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding="utf-8")
        
        # 初始化路径配置
        # self.raw_data_root = self.config.get("Paths", "raw_data_root")
        # self.output_root = self.config.get("Paths", "pipeline_output")
        # self.intermediate_dir = Path(self.output_root) / "intermediate"
        
        # 创建必要目录
        # FileUtil.ensure_directory_exists(self.intermediate_dir)
        # FileUtil.ensure_directory_exists(self.output_root)

    def run_filter(self):
        """执行数据过滤阶段"""
        print("\n" + "="*40 + "\nRunning Filter Stage\n" + "="*40)

        datafilter = DataFilter(self.config)
        datafilter.process_files()
        return str(self.intermediate_dir / "filtered")

    def run_similarity(self,input_file,mode):
        """执行相似度计算"""
        print("\n" + "="*40 + "\nRunning Similarity Calculation\n" + "="*40)
        processor = OptimizedProcessor(self.config,input_file)
        similar_output_path = processor.run(mode)
        
        return similar_output_path

    def run_entity_selection(self, input_dir,mode):
        """执行实体筛选"""
        print("\n" + "="*40 + "\nRunning Entity Selection\n" + "="*40)
        if mode == "High":
            sh_input_file = str(Path(input_dir) / "Special_High.csv")
            sh_output_file = str(Path(input_dir) /"sh_results_selected")
            process_medical_terms(self.config,sh_input_file, sh_output_file, batch_size=30)
        else:
            sl_input_file = str(Path(input_dir) / "Special_Low.csv")
            sl_output_file = str(Path(input_dir) /"sl_results_selected")
            process_medical_terms(self.config,sl_input_file, sl_output_file, batch_size=30)
        
        return input_dir

    def run_concept_fusion(self, change_records):
        """执行概念融合"""
        print("\n" + "="*40 + "\nRunning Concept Fusion\n" + "="*40)
        ori_entities_path = self.config.get("Cal_Similarity","input_file")
        output_name = self.config.get("Fusion","entity_output")
        FileUtil.ensure_directory_exists(output_name)
        fusion = ConceptFusion(change_records, ori_entities_path, output_name)
        fusion.load_data()
        change_path = fusion.process_fusion()
        print("处理完成！")


        return change_path  # 返回变更记录文件路径

    def run_relation_update(self, change_file):
        """更新关系表"""
        print("\n" + "="*40 + "\nUpdating Relations\n" + "="*40)

        data_dir = str(Path(self.config.get("Cal_Similarity","root_path"))/ "Relations")
        out_dir = str(Path(self.config.get("Fusion","output_root"))/"Relations/")
        
        FileUtil.ensure_directory_exists(out_dir)
        update_relations(
            change_file=change_file,
            change_type=self.config.get("Fusion","change_type"),
            data_dir=data_dir,
            out_put_dir=out_dir
        )

    def sep_concept(self):
        root_path = self.config.get("DEFAULT", "root_path")

        ori_entities_path = self.config.get("Cal_Similarity", "input_file")
        processor = ConceptSeparator(
        root_path = root_path,
        entities_path = ori_entities_path,
        )
        out_file, sep_result = processor.process_separation()

        return out_file

    def execute_pipeline(self):
        """执行完整流水线"""
        try:
            #分成两个大阶段，首先是相同概念词的拆分
            #1.相似度计算
            ori_entities_path = self.config.get("Cal_Similarity", "input_file")
            similarity_path = self.run_similarity(ori_entities_path,"Low")
            #2.大模型实体对判断
            change_records = self.run_entity_selection(similarity_path,"Low")
            #3.概念拆分
            sep_entity_file = self.sep_concept()

            #第二个阶段是对不同概念词的融合
            #1.相似度计算
            similarity_path = self.run_similarity(sep_entity_file,"High")
            #2.大模型实体对判断
            change_records = self.run_entity_selection(similarity_path,"High")
            #3.概念融合
            change_file = self.run_concept_fusion(similarity_path)


            # # 阶段1: 数据过滤
            # #filtered_path = self.run_filter()
            #
            # # 阶段2: 相似度计算
            # similarity_path = self.run_similarity()
            # #similarity_path = "/home/dxy/ProjectCode/TaslyAlignment/raw_data/知识图谱_v0318/全量级知识图谱_v0318/实体相似度/疾病实体表_v1.8"
            # # 阶段3: 实体选择
            # change_records = self.run_entity_selection(similarity_path)
            #
            # # 阶段4: 概念融合
            # change_file = self.run_concept_fusion(similarity_path)
            #
            # # 阶段5: 关系更新
            # self.run_relation_update(change_file)
            #
            # print("\n" + "="*40 + "\nPipeline Completed Successfully!\n" + "="*40)
            
        except Exception as e:
            print(f"\nPipeline Failed: {str(e)}")
            raise

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.execute_pipeline()