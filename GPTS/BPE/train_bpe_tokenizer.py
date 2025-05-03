from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers import normalizers

# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([Lowercase()])
tokenizer.pre_tokenizer = Whitespace()

# Trainer settings
trainer = BpeTrainer(
    vocab_size=5000,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)

# Train on your input.txt
tokenizer.train(files=["input.txt"], trainer=trainer)

# Save tokenizer for later use
tokenizer.save("bpe_tokenizer.json")
