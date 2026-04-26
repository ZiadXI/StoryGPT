from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

"""
The tokenizer training is perfomed on kaggle

Make sure to download the dataset on kaggle (4GB+)

"""
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=16384,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    min_frequency=2, # a pair must appear at least twice to be merged
)


if __name__ == "__main__":

 tokenizer.train_from_iterator((example["text"] for example in dataset), trainer=trainer)
 tokenizer.save("storygpt_tokenizer.json")  # saves everything to one file

 text = "boy fun I"

 encoded = tokenizer.encode(text)
 decoded = tokenizer.decode(encoded.ids)
 print("Tokens:", encoded.tokens)
 print("Decoded:", decoded)
