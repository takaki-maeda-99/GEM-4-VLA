"""Placeholder token IDs and Gemma4 metadata.

ID ranges are sub-slices of Gemma4's 6227 unused tokens (258884..262143)
and are kept disjoint so the input packer can identify each block by ID
membership alone. See docs/architectures/x_vla_adapter.md for layout.
"""

# === Gemma4 native image token (PaliGemma-style scene placeholder) ===
# tokenizer.convert_tokens_to_ids('<image_soft_token>') in Gemma4
IMAGE_SOFT_TOKEN_ID: int = 262144 - 1  # confirmed at runtime in Task 11

# === Action queries (carried from vla-gemma-4) ===
ACTION_TOKEN_BEGIN_IDX: int = 258885   # <unused2968>
NUM_ACTION_TOKENS: int = 64

# === Wrist patches ===
WRIST_PLACEHOLDER_BEGIN_IDX: int = 258949  # <unused3032>
NUM_WRIST_TOKENS: int = 256

# === Soft prompt ===
SOFT_PROMPT_BEGIN_IDX: int = 259461 + 1   # one past PROPRIO range from vla-gemma-4
NUM_SOFT_PROMPT_TOKENS: int = 32

# === Architecture-wide defaults (overridable in config) ===
LLM_HIDDEN_DIM: int = 1536
NUM_LLM_LAYERS: int = 35
PLE_DIM: int = 256

NUM_SCENE_TOKENS: int = 256
SIGLIP_HIDDEN_DIM: int = 1152
SIGLIP_IMAGE_SIZE: int = 224

DEFAULT_PROMPT_MAX_LEN: int = 50

ACTION_CHUNK_LEN: int = 8
ACTION_DIM: int = 7
PROPRIO_DIM: int = 8
