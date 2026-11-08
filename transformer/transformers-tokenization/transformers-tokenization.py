class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        vocab_list = []
        for text in texts:
            vocab_list += text.lower().split()
        vocab_set = set(vocab_list)
        vocab_unique_list = ['<PAD>','<UNK>','<BOS>','<EOS>'] + sorted(list(vocab_set))
        counter = 0
        self.vocab_size = len(vocab_unique_list)
        for i in vocab_unique_list:
            self.id_to_word[counter] = i
            self.word_to_id[i] = counter
            counter += 1

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        encoded_list = []
        for i in text.lower().split():
            if i not in self.word_to_id:
                i = self.unk_token
            encoded_list.append(self.word_to_id[i])
        return encoded_list
            

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        decoded_string = ""
        for i in ids:
            if decoded_string != "":
                decoded_string += " "
            if i not in self.id_to_word:
                i = 1
            decoded_string += self.id_to_word[i]
        return decoded_string