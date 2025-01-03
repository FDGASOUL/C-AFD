import os
import time
from Cafd import CAFD


# TODO: 时间不要最后结束，应该在汇总找完所有搜索空间后，输出总时间
class Config:
    DATASET_CONFIG = {
        "IRIS": {"path": "iris", "has_header": False},
        "DATA": {"path": "data", "has_header": True},
        "BEERS": {"path": "beers", "has_header": True},
        "FLIGHTS": {"path": "flights", "has_header": True},
        "HOSPITAL": {"path": "hospital", "has_header": True},
        "ALARM": {"path": "alarm", "has_header": True},
        "ASIA": {"path": "asia", "has_header": True},
    }

    def __init__(self, dataset):
        self.dataset = dataset
        self.input_file_separator = ","
        self.input_folder_path = os.path.join("data", "")
        self.input_file_ending = ".csv"
        self._set_dataset(dataset)
        self.sample_size = 5000
        self.is_null_equal_null = True

    def _set_dataset(self, dataset):
        config = self.DATASET_CONFIG.get(dataset)
        if not config:
            valid_datasets = ", ".join(self.DATASET_CONFIG.keys())
            raise ValueError(f"Unsupported dataset: {dataset}. Valid options are: {valid_datasets}")
        self.input_dataset_name = config["path"]
        self.input_file_has_header = config["has_header"]

    def __str__(self):
        return f"Config:\n\tdataset: {self.input_dataset_name}{self.input_file_ending}"


def execute(config):
    print("Starting CAFD execution...")
    cafd = CAFD(config)
    cafd.execute()


if __name__ == "__main__":
    start_time = time.time()
    conf = Config("ASIA")
    print(conf)
    execute(conf)
    print(f"Total runtime: {int((time.time() - start_time) * 1000)} ms")
