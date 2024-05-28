from typing import Sequence, Dict, Union

import numpy as np
import torch.utils.data as data

from utils.file import load_file_list
import pickle


class Stage3(data.Dataset):
    
    def __init__(
        self,
        file_list: str
    ) -> "Stage3":
        super(Stage3, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        with open(gt_path,'rb') as f:
            return_dict = pickle.load(f)
        return return_dict
    
    def __len__(self) -> int:
        return len(self.paths)
