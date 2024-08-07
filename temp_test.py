from transformers import DistilBertTokenizer, DistilBertModel
import torch

# 假设原始 tensor
tensor_2d = torch.tensor([[1, 2, 3, 256]])  # 用具体的数值替换

# 将所有元素相加
result = torch.sum(tensor_2d)

# 将结果转换为一维 tensor
result_1d = result.unsqueeze(0)  # 变成 1维 tensor

print(result_1d)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
