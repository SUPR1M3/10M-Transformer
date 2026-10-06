class Tokenizer:
    def __init__(self):
        self.token_to_id = {
            "0":0,
            "1":1,
            "2":2,
            "3":3,
            "4":4,
            "5":5,
            "6":6,
            "7":7,
            "8":8,
            "9":9,
            "+":10,
            "=":11,
            " ":12,
        }
        self.id_to_token = {
            0:"0",
            1:"1",
            2:"2",
            3:"3",
            4:"4",
            5:"5",
            6:"6",
            7:"7",
            8:"8",
            9:"9",
            10:"+",
            11:"=",
            12:" ",
        }
        self.vocab_size = len(self.token_to_id)

    def encode(self, text):
        return [self.token_to_id[i] for i in text]

    def decode(self, tokens):
        return "".join(self.id_to_token[int(i)] for i in tokens)

        