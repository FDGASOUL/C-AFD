import os
import random
import pandas as pd


class Sampler:
    def __init__(self, config):
        """
        初始化 Sampler 类，负责数据集的随机抽样。
        :param config: Config 配置对象，包含数据路径及其他配置。
        """
        self.config = config
        self.input_file_path = os.path.join(
            self.config.input_folder_path,
            self.config.input_dataset_name + self.config.input_file_ending,
        )

        self.sample_size = self.config.sample_size

    def random_sample(self):
        """
        对输入数据集进行简单随机抽样。
        """
        print(f"Reading dataset from: {self.input_file_path}")

        # 使用 pandas 读取数据
        try:
            data = pd.read_csv(
                self.input_file_path,
                sep=self.config.input_file_separator,
                header=0 if self.config.input_file_has_header else None,
            )
        except Exception as e:
            raise ValueError(f"Failed to read the input dataset: {e}")

        # 检查抽样数量是否合理
        # TODO: 对于接近抽样数量的数据集，是否还需要进行抽样？
        if self.sample_size > len(data):
            print(f"Sample size ({self.sample_size}) exceeds dataset size ({len(data)}). Returning the entire dataset.")
            return data

        # 随机抽样
        sampled_data = data.sample(n=self.sample_size, random_state=random.randint(0, 10000))
        print(f"Sampled {self.sample_size} rows from the dataset.")

        return sampled_data

    def save_sample(self, output_path="sampled_data.csv"):
        """
        保存抽样后的数据到文件。
        :param output_path: 保存路径，默认保存为 `sampled_data.csv`。
        """
        sampled_data = self.random_sample()
        sampled_data.to_csv(output_path, index=False)
        print(f"Sampled data saved to: {output_path}")