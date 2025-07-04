import os
import random
# from dataclasses import dataclass
from typing import List, Tuple
import json
import torch
from .utils import Rank0ProcessBar

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_file=None, 
        data_dir=None,
        reference_file=None
    ):
        """
            Args:
                data_file： 单个数据文件
                data_dir：文件夹，该文件夹下的所有文件都是数据文件
                reference_file：该文件的每一行存储了一个数据文件路径和该数据文件需要采样的数据量，以`\t`隔开。若数据量为-1，则取全量数据。
        """
        assert not (data_file is None and data_dir is None and reference_file is None)
        data_info = []
        if data_file is not None:
            assert data_dir is None and reference_file is None
            data_info.append((data_file, -1))

        if data_dir is not None:
            assert data_file is None and reference_file is None
            for fname in os.listdir(data_dir):
                data_info.append((os.path.join(data_dir, fname), -1))
        
        if reference_file is not None:
            with open(train_data_file, 'r') as f:
                for line in f:
                    data_file, data_size = line.strip().split('\t')
                    data_size = int(data_size)
                    data_info.append((data_file, data_size))
        
        pbar = Rank0ProcessBar(desc='Loading data', smoothing=0)

        self.dataset = []
        for data_file, data_size in data_info:
            start_index = len(self.dataset)
            if data_size == -1: # 全部都需要添加并训练
                for example in self._load_data(data_file):
                    self.dataset.append(example)
                    pbar.update(1)
            else: # 蓄水池采样
                self.dataset.extend([None for _ in range(data_size)])
                for i, example in enumerate(self._load_data(data_file)):
                    if i < data_size:
                        self.dataset[start_index + i] = example
                    else:
                        if random.random() < data_size / (i + 1): # 替换元素
                            self.dataset[start_index + random.randint(0, data_size - 1)] = example
                for i in range(start_index, len(self.dataset)): # 验证是否正确，第二次运行时可删除
                    pbar.update(1)
                    try:
                        assert self.dataset[i] is not None
                    except Exception:
                        raise RuntimeError(data_file)    
        pbar.close()

    def _load_data(self, fname):
        raise NotImplementedError("需要自定义数据载入流程")
        with open(fname, 'r') as f:
            for line in f:
                line = json.loads(line)
                yield line # (sentence1, sentence2, score)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[str, List[str]]:
        return self.dataset[idx]

class Collator:

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, raw_batch):
        raise NotImplementedError("需要自定义batching流程")
        scores = []
        sentences = []
        for item in raw_batch:
            scores.append(item['score'])
            sentences.append(item['sentence1'])
            sentences.append(item['sentence2'])
        
        encoding = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        batch = {
            'input_ids': encoding.input_ids,
            'attention_mask': encoding.attention_mask,
            'scores': torch.tensor(scores)
        }
        return batch

class LMCollator:

    def __init__(
        self,
        tokenizer,
        user_prompt_template,
        output_template,
        system_prompt: str = None,
        max_length: int = 512,
        add_eos_token: bool = False # 废弃
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.output_template = output_template
        self.add_eos_token = add_eos_token

        self.not_verbose = True

    def __call__(self, raw_batch):

        pairs = []
        for item in raw_batch:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_template.format(**item)}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            output = self.output_template.format(**item)
            pairs.append([prompt, output])

        if self.not_verbose and dist.get_rank() == 0:
            print(pairs[0])
            self.not_verbose = False

        encoding = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=True
        )

        labels = torch.where(
            encoding.token_type_ids.bool(),
            encoding.input_ids, -100
        )
        # if dist.get_rank() == 0:
        #     print((labels >= 0).long().sum(-1))
        
        batch = {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "labels": labels
        }
        return batch
