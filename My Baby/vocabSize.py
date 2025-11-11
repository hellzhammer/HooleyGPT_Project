from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("Datasets/bpe_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size from tokenizer: {vocab_size}")
