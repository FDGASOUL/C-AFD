import os
from BifGenerator import BIFGenerator
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling


class FDDataGeneratorConfig:
    def __init__(self,
                 num_tuples=1000,  # 默认元组数量
                 num_attributes=20,  # 默认属性数量
                 error_rate=0.1,  # 默认错误率
                 domain_size=4,  # 默认域大小
                 input_path_fd='input_fd.txt',  # 默认FD输入路径
                 output_path_csv='data.csv'):  # 默认数据集输出路径

        self.num_tuples = num_tuples
        self.num_attributes = num_attributes
        self.error_rate = error_rate
        self.domain_size = domain_size
        self.input_path_fd = input_path_fd
        self.output_path_csv = output_path_csv
        self.fd_rules = []  # 用来存放解析后的FD规则

    def run(self):
        print("######################打印配置######################")
        print(config)
        # Step 1: 验证参数
        self.validate_params()

        # Step 2: 解析 FD 规则
        self.fd_rules = self.parse_fd_rules(self.input_path_fd)

        # # Step 3: 生成 BIF 文件
        self.generate_bif_file()

        # Step 4: 生成数据集
        self.generate_dataset()

    def __str__(self):
        return (f'FDDataGeneratorConfig(\n'
                f'  num_attributes={self.num_attributes},\n'
                f'  error_rate={self.error_rate},\n'
                f'  domain_size={self.domain_size},\n'
                f'  input_path_fd="{self.input_path_fd}",\n'
                f'  output_path_csv={self.output_path_csv},\n'
                f')')

    def validate_params(self):
        """验证配置参数是否正确"""
        assert self.num_tuples > 0, "num_tuples 应该大于 0"
        assert self.num_attributes > 0, "num_attributes 应该大于 0"
        assert 0 <= self.error_rate <= 1, "error_rate 应该在 0 到 1 之间"
        assert self.domain_size > 0, "domain_size 应该大于 0"
        assert os.path.exists(self.input_path_fd), f"输入的FD文件路径不存在: {self.input_path_fd}"
        print("参数验证通过！")

    def parse_fd_rules(self, path=None):
        """从FD文件中解析规则"""
        fd_rules = []
        file_path = path if path else self.input_path_fd
        with open(file_path, 'r') as fd_file:
            lines = fd_file.readlines()
            for line in lines:
                line = line.strip()
                if '->' in line:
                    lhs, rhs = line.split('->')
                    lhs = [str(int(x.strip().replace("column", "")) - 1) for x in lhs.split(',')]
                    rhs = str(int(rhs.strip().replace("column", "")) - 1)
                    fd_rules.append((lhs, rhs))
        print(f"成功解析FD规则: {fd_rules}")
        return fd_rules

    def generate_bif_file(self):
        """根据FD生成BIF文件"""
        bif_file_path = self.output_path_csv.replace('.csv', '.bif')  # 假设bif文件路径
        bif_generator = BIFGenerator(self.num_attributes, self.fd_rules, bif_file_path, self.error_rate, domain_sizes={
                'column1': 5,
                'column2': 3,
                'column3': 5,
                'column4': 4,
                'column5': 5,
                'column6': 4,
                'column7': 4,
                'column8': 5,
                'column9': 4,
                'column10': 3,
                'column11': 4,
                'column12': 5,
                'column13': 3,
                'column14': 6,
                'column15': 3,
                'column16': 3,
                'column17': 5,
                'column18': 7,
                'column19': 4,
                'column20': 4,
                'column21': 4,
                'column22': 4,
                'column23': 4,
                'column24': 4,
                'column25': 4,
                'column26': 4,
                'column27': 4,
                'column28': 4,
                'column29': 4,
                'column30': 4,
                'column31': 4,
                'column32': 4,
                'column33': 4,
            })
        bif_generator.generate_bif()

    def generate_dataset(self):
        """生成数据集"""
        bif_file_path = self.output_path_csv.replace('.csv', '.bif')  # 假设bif文件路径
        # Step 1: 读取BIF文件并构建贝叶斯网络模型
        reader = BIFReader(bif_file_path)
        model = reader.get_model()

        # Step 2: 使用pgmpy进行采样
        inference = BayesianModelSampling(model)
        samples = inference.forward_sample(size=self.num_tuples)

        # Step 3: 将生成的数据集保存为CSV
        samples.to_csv(self.output_path_csv, index=False)
        print(f"数据集已生成并保存到: {self.output_path_csv}")


if __name__ == "__main__":
    config = FDDataGeneratorConfig(
        num_tuples=10000,
        num_attributes=33,
        error_rate=0.1,
        domain_size=5,
        input_path_fd='data_generator/data/input_fd.txt',
        output_path_csv='data_generator/data/data.csv',
    )

    config.run()
