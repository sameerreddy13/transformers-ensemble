from pathlib import Path
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from datasets import load_dataset
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# dataset_wiki = load_dataset("wikitext", "wikitext-2")
# dataset_book = load_dataset("bookcorpus")

# Special Tokens
unk_token = "[UNK]"  # token for unknown words
spl_tokens = [unk_token, "[CLS]", "[SEP]", "[PAD]", "[MASK]"]  # special tokens

# Initialize a tokenizer
tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

trainer = WordPieceTrainer(
    vocab_size=30522,
    special_tokens=spl_tokens,  # TODO: Ensure vocab size matches the joined datasets
)
# TODO: Add code to download and unzip the data, if desired
files = [
    f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]
]
tokenizer.train(files, trainer)
tokenizer.save("data/bert-wiki.json")