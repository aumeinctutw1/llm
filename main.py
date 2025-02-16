import tiktoken
import torch
import GPTDatasetV1

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                        stride=128, shuffle=True,
                        drop_last=True, num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1.GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
    

if __name__ == "__main__":
    print(f"Torch backend mps available: {torch.backends.mps.is_available()}")

    with open('the-verdict.txt', 'r', encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)
