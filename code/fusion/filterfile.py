import pandas as pd
import re
import os
from configparser import ConfigParser

class DataFilter:
    """Excel数据过滤处理器"""
    def __init__(self, config):
        self.config = config
        
        # 从配置文件读取参数
        self.input_path = self.config.get('FilterFile', 'input_path')
        self.out_root = self.config.get('FilterFile', 'output_root')
        self.filter_columns =['SubjectName','ObjectName'] #对于实体为['Name']，对于关系为['SubjectName','ObjectName']
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')  # 预编译正则

    def process_files(self):
        """执行完整处理流程"""
        for filename, filepath in self._get_all_filenames():
            rel_path = os.path.relpath(os.path.dirname(filepath), self.input_path)
            output_dir = os.path.join(self.out_root, rel_path)
            os.makedirs(output_dir, exist_ok=True)  # 自动创建目录
            
            output_path = os.path.join(output_dir, filename)
            self._filter_english_rows(filepath, output_path)
        return self.out_root

    def _filter_english_rows(self, input_file, output_file):
        """核心过滤逻辑"""
        try:
            df = pd.read_excel(input_file)
            # 多列联合过滤
            mask = pd.Series([True] * len(df))
            for col in self.filter_columns:
                mask &= df[col].apply(lambda x: not self._contains_chinese(str(x)))
            
            filtered_df = df[mask]
            
            if not filtered_df.empty:
                filtered_df.to_excel(output_file, index=False)
                print(f"保存成功: {output_file}")
            else:
                print(f"无英文数据: {input_file}")
        except Exception as e:
            print(f"处理失败 {input_file}: {str(e)}")

    def _contains_chinese(self, text):
        """中文字符检测"""
        return bool(self.chinese_pattern.search(text))

    def _get_all_filenames(self):
        """递归获取所有Excel文件"""
        for root, _, files in os.walk(self.input_path):
            for file in files:
                if file.endswith(('.xlsx', '.xls')):
                    yield os.path.basename(file), os.path.join(root, file)

if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini", encoding="utf-8")
    processor = DataFilter(config)
    processor.process_files()
