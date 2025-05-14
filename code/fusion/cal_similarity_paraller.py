#time:2025/03/13
#function:多进程计算实体之间的语义相似度
import sys
import os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from configparser import ConfigParser

import multiprocessing as mp
from sklearn.metrics.pairwise import pairwise_distances
import math
import gc
from scipy.sparse import csr_matrix

from util.FileUtil import FileUtil




class OptimizedProcessor:
    def __init__(self, config,input_file):
        self.cfg = config
        self.bin_size = self.cfg.getint("Cal_Similarity","bin_size")
        self.bins = self._generate_bins()
        self.bin_edges = np.array([0] + [b[1] for b in self.bins])
        
        self.batch_size = self.cfg.getint("Cal_Similarity","batch_size")
        self.special_low = self.cfg.getint('Cal_Similarity',"special_low")
        self.special_high = self.cfg.getint('Cal_Similarity',"special_high")
        self.model_name = self.cfg.get('Cal_Similarity','model_name')
        # self.input_file = self.cfg.get('Cal_Similarity','input_file')
        self.input_file = input_file
        self.output_file = self.cfg.get('Cal_Similarity','output_file')
        self.result_file = self.cfg.get('Cal_Similarity','diff_result_file')
        self.embedding_file = self.cfg.get('Cal_Similarity',"embedding_file")
        self.shared = {
            'embeddings': None,
            'concept_ids': None,
            'atom_ids': None,
            'names': None
        }

    def _generate_bins(self):
        """生成区间配置"""
        return [(i, i+self.bin_size) 
               for i in range(0, 100, self.bin_size)]

    def _initialize_shared_data(self):
        """初始化共享数据（无文件存储）"""
        df = pd.read_excel(self.input_file)
        self.df = df
        # 转换为numpy数组
        self.shared['concept_ids'] = df['ConceptId'].values
        self.shared['atom_ids'] = df['AtomId'].values
        self.shared['names'] = df['Name'].values.astype('U')  # 统一为Unicode
        
        # 生成嵌入向量
        FileUtil.ensure_directory_exists(self.embedding_file)
        if not os.path.exists(self.embedding_file):
            model = SentenceTransformer(self.model_name)
            embeddings = model.encode(df['Name'].tolist(), 
                                    show_progress_bar=True,
                                    batch_size=self.batch_size)  # 优化推理批次
            
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            print(embeddings.shape)

            with open(self.embedding_file, "wb") as f:
                pickle.dump(embeddings, f)
            
        
        # 内存映射方式加载大文件
        self.shared['embeddings'] = np.load(
            self.embedding_file, 
            mmap_mode='r',allow_pickle=True
        )

    def _get_concept_groups(self):
        """生成需要处理的Concept分组"""
        unique_concepts, inverse = np.unique(
            self.shared['concept_ids'], return_inverse=True
        )
        counts = np.bincount(inverse)
        return unique_concepts[counts > 1]  # 仅处理包含多个实体的Concept
    
    def _batch_cosine_similarity(self, mat1, mat2):
        """批量计算余弦相似度"""
        # 归一化向量
        mat1_norm = mat1 / np.linalg.norm(mat1, axis=1, keepdims=True)
        mat2_norm = mat2 / np.linalg.norm(mat2, axis=1, keepdims=True)
         # 计算余弦相似度
        cos_sim = np.dot(mat1_norm, mat2_norm.T)
        # 映射到[0, 100]
        cos_sim = (cos_sim + 1) * 50
        cos_sim = cos_sim.astype(int)
        return cos_sim
    
    def _get_interval(self, sim):
        """获取相似度对应的区间标签"""
        try:
            for lower, upper in self.bins:
                if lower <= sim <= upper:
                    return f"{lower}-{upper}"
        except ValueError as e:
            print("引发异常：",repr(e))
            print(sim)
        return sim
    
    def process_same_concept(self):
        """处理相同ConceptId"""
        results = {f"{l}-{u}": [] for l, u in self.bins}
        special_low = []
        print(results)
        for concept_id, group in tqdm(
            self.df.groupby('ConceptId'),
            desc="处理相同Concept",
            mininterval=self.cfg.getfloat('Cal_Similarity',"progress_refresh")
        ):
            indices = group.index.tolist()
            batch_embeddings = self.shared['embeddings'][indices]
            sim_matrix = self._batch_cosine_similarity(
                batch_embeddings, batch_embeddings)
            
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    sim = sim_matrix[i, j]
                    
                    # 获得相同conceptid但是相似度较低的实体
                    if sim < self.special_low:
                        special_low.append({
                            "ConceptId": concept_id,
                            "AtomId1": group.iloc[i]['AtomId'],
                            "AtomId2": group.iloc[j]['AtomId'],
                            "Name1":group.iloc[i]['Name'],
                            "Name2":group.iloc[j]['Name'],
                            "Similarity": sim
                        })
                    
                    # 常规区间记录
                    
                    interval = self._get_interval(sim)
                    results[interval].append({
                        "ConceptId": concept_id,
                        "AtomId1": group.iloc[i]['AtomId'],
                        "AtomId2": group.iloc[j]['AtomId'],
                        "Name1":group.iloc[i]['Name'],
                        "Name2":group.iloc[j]['Name'],
                        "Similarity": sim
                    })
        #保存文件
        output_dir = self.output_file
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir,"Special_Low.csv")
        pd.DataFrame(special_low).to_csv(file_path)

        #删除不用数据，节省内存
        del self.df
        gc.collect()
        return results, special_low

    def _process_diff_chunk(self, chunk_start):
        """处理不同Concept的分块计算"""
        chunk_size = self.batch_size
        chunk_end = min(chunk_start + chunk_size, len(self.shared['embeddings']))
        indices = np.arange(chunk_start, chunk_end)
        
        # 分块计算相似度
        embeddings = self.shared['embeddings'][indices]
        sim_matrix = (1 - pairwise_distances(
            embeddings, 
            self.shared['embeddings'],
            metric='cosine'
        )) * 100
        sim_matrix = sim_matrix.astype(np.int8)
        
        # 生成候选对
        # rows, cols = np.where(sim_matrix >= self.special_high)  # 过滤0相似度
        # valid_mask = (indices[rows] < cols)  # 避免重复计数
        # return {
        #     'pairs': np.column_stack((indices[rows][valid_mask], cols[valid_mask])),
        #     'sims': sim_matrix[rows, cols][valid_mask]
        # }
        sparse_sim = csr_matrix(sim_matrix)
        sparse_sim.data[sparse_sim.data < self.special_high] = 0
        sparse_sim.eliminate_zeros()  # 清除零元素，減少存储

        # 获取非零元素的位置（即>=special_high的坐标）
        rows, cols = sparse_sim.nonzero()
        valid_mask = (indices[rows] < cols)  # 避免重复对

        # 构建结果
        return {
            'pairs': np.column_stack((indices[rows][valid_mask], cols[valid_mask])),
            'sims': sparse_sim.data[valid_mask]  # 直接使用稀疏矩阵中的值
        }

    def _classify_results(self, pairs, sims, is_same_concept):
        """分类计算结果"""
        # 区间分类
        bins = np.digitize(sims, self.bin_edges) - 1
        valid_bins = (bins >= 0) & (bins < len(self.bins))
        
        # 特殊条件筛选
        if is_same_concept:
            special_mask = sims < self.special_low
        else:
            special_mask = sims >= self.special_high
        
        return {
            'bins': bins[valid_bins],
            'pairs': pairs[valid_bins],
            'sims': sims[valid_bins],
            'special': pairs[special_mask & valid_bins],
            'special_sims': sims[special_mask & valid_bins]
        }
    
    def _save_results_chunk(self, results, is_same_concept, header):
        """优化后的结果保存方法（CSV版）"""
        output_dir = self.output_file
        os.makedirs(output_dir, exist_ok=True)
        chunk_size = 50000  # 根据内存调整分块大小

        # 保存特殊结果
        if len(results['pairs']) > 0:
            special_pairs = results['pairs']
            special_sims = results['sims']
            total = len(special_pairs)
            
            prefix = "Special_Low" if is_same_concept else "Special_High"
            filename = f"{prefix}.csv"
            file_path = os.path.join(output_dir, filename)
            
            for chunk_start in range(0, total, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total)
                chunk_pairs = special_pairs[chunk_start:chunk_end]
                chunk_sims = special_sims[chunk_start:chunk_end]

                df = self._create_result_chunk(chunk_pairs, chunk_sims, is_same_concept)
                # 生成排序后的唯一键（元组或字符串）
                df['key'] = df.apply(lambda row: tuple(sorted([row['Name1'], row['Name2']])), axis=1)
                # 去重（保留首次出现的行）
                df = df.drop_duplicates(subset='key', keep='first')
                # 清理临时列并保存结果
                df.drop('key', axis=1, inplace=True)
                
                df.to_csv(
                    file_path,
                    mode="w" if header else "a",
                    index=False,
                    encoding='UTF-8',
                    header=header
                )

    def _create_result_chunk(self, pairs_chunk, sims_chunk, is_same_concept):
        """优化后的分块DataFrame生成"""
        # 使用numpy视图避免内存拷贝
        idx1 = pairs_chunk[:, 0]
        idx2 = pairs_chunk[:, 1]
        
        # 仅在不同Concept时需要过滤相同ConceptId的对
        if not is_same_concept:
            # 提前计算Concept差异掩码（避免重复计算）
            concept_ids = self.shared['concept_ids']  # 预加载减少查找次数
            mask = concept_ids[idx1] != concept_ids[idx2]
            
            # 应用掩码过滤（原地操作减少内存分配）
            idx1 = idx1[mask]
            idx2 = idx2[mask]
            sims_chunk = sims_chunk[mask]

        # 直接构建字典（避免中间变量）
        data = {
            'AtomId1': self.shared['atom_ids'][idx1],
            'AtomId2': self.shared['atom_ids'][idx2],
            'Name1': self.shared['names'][idx1],
            'Name2': self.shared['names'][idx2],
            'Similarity': sims_chunk
        }
        
        # 添加Concept信息
        if is_same_concept:
            data['ConceptId'] = self.shared['concept_ids'][idx1]
            return pd.DataFrame(data).astype({
            'AtomId1': 'category',
            'AtomId2': 'category',
            'ConceptId': 'category',
            'Name1': 'category',
            'Name2': 'category',
            
            })
        else:
            data['ConceptId1'] = self.shared['concept_ids'][idx1]
            data['ConceptId2'] = self.shared['concept_ids'][idx2]
            # 使用低内存模式创建DataFrame
            return pd.DataFrame(data).astype({
                'AtomId1': 'category',
                'AtomId2': 'category',
                'ConceptId1': 'category' ,
                'ConceptId2': 'category' ,
                'Name1': 'category',
                'Name2': 'category',
                
            })


    def process_diff_concept(self):
        '''处理不同Concept'''
        # 第一阶段：计算所有相似度
        
        print("Phase 1: Computing similarities...")
        try:
            with open(self.result_file, "rb") as f:
                diff_results = pickle.load(f) #是一个由多线程处理结果合并而成的字典嵌套list
            print("发现已缓存的相似度文件，跳过计算步骤")
        except FileNotFoundError:
            diff_results = []
            total = len(self.shared['embeddings'])
            chunks = range(0, total, self.cfg.getint("Cal_Similarity","batch_size"))
            with mp.Pool(processes=mp.cpu_count()) as pool:
                diff_results = list(tqdm(
                    pool.imap(self._process_diff_chunk, chunks),
                    total=len(chunks),
                    desc="Diff Concept Processing"
                ))
            with open(self.result_file, "wb") as f:
                    pickle.dump(diff_results, f)

        # 第二阶段：分类保存结果
        print("\nPhase 2: Classifying and Saving results...")
        print(type(diff_results))
        
        header = True
        for diff_dict in tqdm(diff_results,total=len(diff_results)):
            self._save_results_chunk(diff_dict, False, header) #创建文件夹保存不同id结果
            header = False

    def run(self,mode):
        """主执行流程"""
        self._initialize_shared_data()

        # 处理相同Concept
        if mode == "High":
            print("处理不同Concept")
            self.process_diff_concept()
        else:
            print("处理相同Concept")
            self.process_same_concept()


        print(f"\n处理完成！结果保存在: {self.output_file}")
        return self.output_file

        
if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini", encoding="utf-8")
    processor = OptimizedProcessor(config)
    processor.run()
