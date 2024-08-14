import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer
import config as CFG
from CLIP import CLIPModel
from train import build_loaders, to_device
from dataset import Load_dataset 


def find_rank(score_list, idx):
    value = score_list[idx]
    sorted_score_list = sorted(score_list)
    rank = sorted_score_list.index(value) + 1
    return rank

def main():
    text_input = "./data/kg_T_self_160.txt"
    image_input = "./data/img"  
    motion_input = "./data/motion"  

    model = CLIPModel()
    model.to(CFG.device)
    model.load_state_dict(torch.load("final.pt"))

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    dataset = Load_dataset(text_input, image_input, motion_input, tokenizer)
    test_loader = build_loaders(dataset)

    len_test = len(dataset)
    tqdm_object = tqdm(test_loader, total=len_test)
    MRR=H10=H1=0
    
    for idx, batch in enumerate(tqdm_object):
        score_list = []
        batch = to_device(batch, CFG.device)
        for data in test_loader:
            data = to_device(data, CFG.device)
            score = model(batch, data)
            score_list.append(score)

        rank = find_rank(score_list,idx)
        MRR += 1/rank

        sorted_indices = sorted(range(len(score_list)), key=lambda i: score_list[i])[:10]
        if idx in sorted_indices:
            H10 += 1

        min_idx = score_list.index(min(score_list))
        if min_idx == idx:
            H1 += 1

    H1 = H1 / len_test
    H10 = H10 / len_test
    MRR = MRR / len_test

    print(f"MRR: {MRR}, H@10: {H10}, H@1: {H1}")


if __name__=='__main__':
    main()