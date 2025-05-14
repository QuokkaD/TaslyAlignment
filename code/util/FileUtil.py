import os

class FileUtil:
    """
    文件操作工具类，提供自动创建目录并保存文件的功能。
    """
    @staticmethod
    def ensure_directory_exists(file_path: str) -> None:
        """
        确保文件路径中的目录存在，如果不存在则创建。

        Args:
            file_path (str): 文件路径，可以是绝对或相对路径。
        """
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
