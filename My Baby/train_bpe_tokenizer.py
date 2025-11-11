import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Lowercase, Strip, Sequence
from tokenizers.pre_tokenizers import ByteLevel

# === CONFIGURATION ===
DATA_DIR = "Datasets"
OUTPUT_FILE = "Datasets/bpe_tokenizer.json"
VOCAB_SIZE = 64_000

# === INIT TOKENIZER ===
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase(),
    Strip()
])

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[
        "[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SEP]", "[COPY]"
    ]
)

# === GET ALL .txt FILES IN THE DIRECTORY ===
txt_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

if not txt_files:
    raise RuntimeError(f"No .txt files found in directory: {DATA_DIR}")

print(f"Found {len(txt_files)} text files for tokenizer training...")

# === TRAIN ===
tokenizer.train(files=txt_files, trainer=trainer)

# === SAVE ===
tokenizer.save(OUTPUT_FILE)
print(f"✅ Tokenizer training complete. Saved to '{OUTPUT_FILE}'")
