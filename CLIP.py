import torch
from torch import nn
import torch.nn.functional as F

from config import CFG
from modules import ImageEncoder, TextEncoder, MotionEncoder, ProjectionHead
import copy
############CLIP############
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
        motion_embedding = CFG.motion_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.motion_encoder = MotionEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.motion_projection = ProjectionHead(embedding_dim=motion_embedding)
        self.temperature = temperature

    def get_text_features(self,or_text):
        '''
        
        '''
        def combine_tensors(u_list):
            sum_bd = sum(tensor[0] for tensor in u_list) / len(u_list)
            sum_tail = sum(tensor[1] for tensor in u_list) / len(u_list)

            return [sum_bd, sum_tail]

        def calculate_summation(triplet_features_list):
            # # 调整数据尺度
            # scale_factor = 1  # 或其他适当的值
            # triplet_features_list = [feat * scale_factor for feat in triplet_features_list]
            # # 计算第一个元素和后7个元素的卷积并相加
            # conv_result = torch.zeros_like(triplet_features_list[0].unsqueeze(0))
            # conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=768, stride=1, padding=0, bias=False)
            # conv.weight.data = triplet_features_list[0].unsqueeze(0)
            # for i in range(2, 9):
            #     # 太小会成0
            #     conv_result += conv(triplet_features_list[i].unsqueeze(0))

            # # 创建新列表new_triplet_features_list
            # new_triplet_features_list = [conv_result]

            # # 将triplet_features_list中的第二个元素添加到new_triplet_features_list中
            # new_triplet_features_list.append(triplet_features_list[1])
            # return new_triplet_features_list
            #////////////////////////////////////////////////////////////////////////////////////////////////////////#
            # 计算第一个元素和后7个元素的卷积并相加
            conv_result = triplet_features_list[0]
            for i in range(2, 9):
                conv_result += triplet_features_list[i]

            # 创建新列表new_triplet_features_list
            different_relation = 5.0
            new_triplet_features_list = [conv_result/different_relation]

            # 将triplet_features_list中的第二个元素添加到new_triplet_features_list中
            new_triplet_features_list.append(triplet_features_list[1])
            return new_triplet_features_list

        text = copy.deepcopy(or_text)
        u_list = []
        for triplet in text:
            triplet_features_list = []
            for i in range(9):
                # 获取当前样本的input_ids和attention_mask
                input_ids = triplet["input_ids"][0,i,:].unsqueeze(0).to(CFG.device)
                attention_mask =  torch.zeros_like(triplet["attention_mask"][0,i,:].unsqueeze(0)).to(CFG.device)
                # 将当前样本传递给text_encoder并获取特征
                entity_or_relation_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                triplet_features_list.append(entity_or_relation_features)
            # 卷积整合三元组
            u_vector = calculate_summation(triplet_features_list)
            u_list.append(u_vector)
        text_vector = combine_tensors(u_list)
        return text_vector
    
    def forward(self, batch):
        batch_text_embeddings = []
        batch_image_embeddings = []
        batch_motion_embeddings = []
        # Getting Image and Text Features 每次处理一个时刻
        for group in range(len(batch['image'])):
            image = batch["image"][group].to(CFG.device)
            text = batch["text"][group]
            motion_index = min(group, len(batch["motion"]) - 1)
            motion = batch["motion"][motion_index].to(CFG.device)
            image_features = self.image_encoder(image)
            text_features = self.get_text_features(text)
            motion_features = self.motion_encoder(motion)

            # Getting Image and Text Embeddings (with same dimension)
            image_embeddings = self.image_projection(image_features)
            #text_embeddings[1]是手语词
            text_embeddings = [self.text_projection(tensor) for tensor in text_features]
            # text_embeddings = self.text_projection(text_features)
            motion_embeddings = self.motion_projection(motion_features)
            print(f"""image_embeddings size: {image_embeddings.size()}\n
                  text_embeddings[0] size: {text_embeddings[0].size()}\n
                  text_embeddings[1] size: {text_embeddings[1].size()}\n
                  motion_embeddings size: {motion_embeddings.size()}""")
            # Calculating the CLIP Loss
            logits = (text_embeddings[0] @ image_embeddings.T) / self.temperature
            images_similarity = image_embeddings @ image_embeddings.T
            texts_similarity = text_embeddings[0] @ text_embeddings[0].T
            targets = F.softmax(
                (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
            )
            texts_loss = cross_entropy(logits, targets, reduction='none')
            images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
            loss =  (images_loss + texts_loss)

            batch_text_embeddings.append(text_embeddings)
            batch_image_embeddings.append(image_embeddings)
            batch_motion_embeddings.append(motion_embeddings)
        # SIGN_loss
        loss_t_i = score_t_i(batch_text_embeddings,batch_image_embeddings)
        loss_m_i = score_m_i(batch_image_embeddings,batch_motion_embeddings)
        sign_loss = CFG.t_i*loss_t_i + CFG.i_m*loss_m_i
        loss += sign_loss
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def score_t_i(batch_text_embeddings,batch_image_embeddings):
    head_embeddings_sum = torch.zeros_like(batch_text_embeddings[0][0]).to(CFG.device)
    tail_embeddings_sum = torch.zeros_like(batch_text_embeddings[0][1]).to(CFG.device)
    image_embeddings_sum = torch.zeros_like(batch_image_embeddings[0]).to(CFG.device)
    for text_embeddings, image_embeddings in zip(batch_text_embeddings, batch_image_embeddings):
        head_embeddings_sum += text_embeddings[0]
        image_embeddings_sum += image_embeddings
        tail_embeddings_sum += text_embeddings[1]
    avg_expression = (head_embeddings_sum + image_embeddings_sum) / 2 - tail_embeddings_sum
    score = torch.norm(avg_expression, p=2).unsqueeze(0)
    # max_term = torch.max(torch.tensor(0), score + CFG.gamma - score_prime)
    max_term = torch.max(torch.tensor([0]), score)
    return max_term

def score_m_i(batch_image_embeddings,batch_motion_embeddings): 
    score = torch.tensor([0])
    length = len(batch_motion_embeddings)
    for k in range(length-1):
        current_s = torch.norm(batch_image_embeddings[k] + batch_motion_embeddings[k] - batch_image_embeddings[k+1], p=2).unsqueeze(0)
        score += torch.max(torch.tensor([0]),current_s)
    return score

# if __name__ == '__main__':
    # # test
    # images = torch.randn(8, 3, 224, 224)
    # input_ids = torch.randint(5, 300, size=(8, 25))
    # attention_mask = torch.ones(8, 25)
    # batch = {
    #     'image': images,
    #     'input_ids': input_ids,
    #     'attention_mask': attention_mask
    # }

    # CLIP = CLIPModel()
    # loss = CLIP(batch)
    # print("")
    