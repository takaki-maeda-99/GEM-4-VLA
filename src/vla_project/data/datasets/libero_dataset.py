import torch
from torch.utils.data import Dataset

from vla_project.data import constants as C


class SyntheticLIBEROBatchDataset(Dataset):
    """Yields random tensors that match the internal Batch schema.

    Used for smoke tests until the real LIBERO reader is hooked up.
    """

    def __init__(self, length: int = 64, prompt_max_len: int = C.DEFAULT_PROMPT_MAX_LEN):
        self.length = length
        self.prompt_max_len = prompt_max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "domain_id": torch.tensor(0, dtype=torch.long),
            "scene_image": torch.randn(3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE),
            "wrist_image": torch.randn(3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE),
            "prompt_input_ids": torch.zeros(self.prompt_max_len, dtype=torch.long),
            "prompt_attention_mask": torch.zeros(self.prompt_max_len, dtype=torch.long),
            "proprio": torch.randn(C.PROPRIO_DIM),
            "last_action_chunk": torch.randn(C.ACTION_CHUNK_LEN, C.ACTION_DIM),
            "target_action": torch.randn(C.ACTION_CHUNK_LEN, C.ACTION_DIM),
            "action_mask": torch.ones(C.ACTION_CHUNK_LEN, dtype=torch.bool),
        }

    @staticmethod
    def collate_fn(samples):
        keys = samples[0].keys()
        return {k: torch.stack([s[k] for s in samples]) for k in keys}
