# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
