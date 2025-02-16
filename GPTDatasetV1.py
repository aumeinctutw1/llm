from torch.utils.data import Dataset

# Is based on the pytorcj Dataset class
# It defines how individual samples are retrieved from the dataset
# where each sample contains of a number of token ids (based on the max_length)
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text
        tokens_ids = tokenizer.encode(txt)

        for i in range(0, len(tokens_ids) - max_length, stride):
            # use sliding window to chunk the text into overlapping sequences of max_length
            input_chunk = tokens_ids[i: i + max_length]
            target_chunk = tokens_ids[i + 1: i + max_length + 1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)
    
    # returns the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)
    
    # returns a single row from the dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]