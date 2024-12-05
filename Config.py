import os
import time


class Config:
    DATASET_CONFIG = {
        "IRIS": {"path": "iris", "has_header": False},
        "DATA": {"path": "data", "has_header": False},
    }

    def __init__(self, dataset):
        self.dataset = dataset
        self.input_file_separator = ","
        self.input_folder_path = os.path.join("data", "")
        self.input_file_ending = ".csv"
        self._set_dataset(dataset)

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
    input_file_path = os.path.join(config.input_folder_path, config.input_dataset_name + config.input_file_ending)
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    ground_truth_path = os.path.join("groundtruth", config.input_dataset_name + ".txt")
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    print(f"Executing algorithm on dataset: {input_file_path}")
    print(f"Ground truth located at: {ground_truth_path}")


if __name__ == "__main__":
    start_time = time.time()
    conf = Config("IRIS")
    print(conf)
    execute(conf)
    print(f"Total runtime: {int((time.time() - start_time) * 1000)} ms")
