import torch
from torch import nn
import torch.nn.functional as F
from config import CFG
from modules import ImageEncoder, TextEncoder, MotionEncoder, ProjectionHead

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
        motion_embedding=CFG.motion_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.motion_encoder = MotionEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.motion_projection = ProjectionHead(embedding_dim=motion_embedding)
        self.temperature = temperature

    def get_text_features(self, or_text):

        def combine_tensors(u_list):
            sum_bd = sum(tensor[0] for tensor in u_list) / len(u_list)
            sum_tail = sum(tensor[1] for tensor in u_list) / len(u_list)
            return [sum_bd, sum_tail]

        def calculate_summation(triplet_features_list,different_relation=5):
            conv_result = triplet_features_list[0]
            for i in range(2, 9):
                conv_result += triplet_features_list[i]

            new_triplet_features_list = [conv_result/different_relation]

            new_triplet_features_list.append(triplet_features_list[1])
            return new_triplet_features_list

        text = or_text
        u_list = []
        different_relation = len(text)
        for triplet in text:
            triplet_features_list = []
            for i in range(9):
                input_ids = triplet["input_ids"][0,i,:].unsqueeze(0)
                attention_mask = torch.zeros_like((triplet["attention_mask"][0,i,:].unsqueeze(0)))
                entity_or_relation_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                triplet_features_list.append(entity_or_relation_features)

            u_vector = calculate_summation(triplet_features_list, different_relation)
            u_list.append(u_vector)
        text_vector = combine_tensors(u_list)
        return text_vector
    
    def forward(self, batch, batch_prime):
        batch_text_embeddings = []
        batch_image_embeddings = []
        batch_motion_embeddings = []

        batch_prime_text_embeddings = []
        batch_prime_image_embeddings = []
        batch_prime_motion_embeddings = []

        for group_prime in range(len(batch_prime['image'])):
            image_prime = batch_prime["image"][group_prime]
            text_prime = batch_prime["text"][group_prime]
            if len(batch_prime["motion"]) == 0:
                motion_prime = torch.zeros((1, 176, 274), device=image_prime.device)
            else:
                motion_index = min(group_prime, len(batch_prime["motion"]) - 1)
                motion_prime = batch_prime["motion"][motion_index]

            image_prime_features = self.image_encoder(image_prime)
            text_prime_features = self.get_text_features(text_prime)
            motion_prime_features = self.motion_encoder(motion_prime)

            image_prime_embeddings = self.image_projection(image_prime_features)
            text_prime_embeddings = [self.text_projection(tensor) for tensor in text_prime_features]
            motion_prime_embeddings = self.motion_projection(motion_prime_features)

            batch_prime_text_embeddings.append(text_prime_embeddings)
            batch_prime_image_embeddings.append(image_prime_embeddings)
            batch_prime_motion_embeddings.append(motion_prime_embeddings)

        for group in range(len(batch['image'])):
            image = batch["image"][group]
            text = batch["text"][group]
            if len(batch["motion"]) == 0:
                motion = torch.zeros((1, 219, 274),device=image.device)
            else:
                motion_index = min(group, len(batch["motion"]) - 1)
                motion = batch["motion"][motion_index]

            image_features = self.image_encoder(image)
            text_features = self.get_text_features(text)
            motion_features = self.motion_encoder(motion)

            image_embeddings = self.image_projection(image_features)
            text_embeddings = [self.text_projection(tensor) for tensor in text_features]
            motion_embeddings = self.motion_projection(motion_features)
            print(f"""image_embeddings size: {image_embeddings.size()}\n
                  text_embeddings[0] size: {text_embeddings[0].size()}\n
                  text_embeddings[1] size: {text_embeddings[1].size()}\n
                  motion_embeddings size: {motion_embeddings.size()}""")

            # # Calculating the CLIP Loss(easy)
            # text_embeddings[0] = F.normalize(text_embeddings[0], p=2, dim=-1)
            # image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            # logits = torch.matmul(text_embeddings[0], image_embeddings.t()) / self.temperature
            # loss = -torch.log(torch.sigmoid(logits)).squeeze(0)
            # print('CLIP loss:', loss)

            # Calculating the CLIP Loss(standard)
            text_embeddings[0] = F.normalize(text_embeddings[0], p=2, dim=-1)
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            logits = torch.matmul(text_embeddings[0], image_embeddings.t()) / self.temperature
            log_probs = F.log_softmax(logits, dim=-1)
            targets = torch.arange(text_embeddings[0].size(0))
            loss = F.nll_loss(log_probs, targets).unsqueeze(0)
            print('CLIP loss:', loss)

            # # Calculating the CLIP Loss(hard)
            # logits = (text_embeddings[0] @ image_embeddings.T) / self.temperature
            # images_similarity = image_embeddings @ image_embeddings.T
            # texts_similarity = text_embeddings[0] @ text_embeddings[0].T
            # targets = F.softmax(
            #     (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
            # )
            # texts_loss = cross_entropy(logits, targets, reduction='none')
            # images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            # loss = (images_loss + texts_loss) / 2.0 # shape: (batch_size)
            # print('CLIP loss:', loss)

            batch_text_embeddings.append(text_embeddings)
            batch_image_embeddings.append(image_embeddings)
            batch_motion_embeddings.append(motion_embeddings)

        # SIGN_loss
        loss_t_i = score_t_i(batch_text_embeddings,batch_image_embeddings,batch_prime_text_embeddings,batch_prime_image_embeddings)
        loss_m_i = score_m_i(batch_image_embeddings,batch_motion_embeddings,batch_prime_image_embeddings,batch_prime_motion_embeddings)
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

def score_t_i(batch_text_embeddings,batch_image_embeddings,batch_prime_text_embeddings,batch_prime_image_embeddings):
    head_embeddings_sum = torch.zeros_like(batch_text_embeddings[0][0])
    tail_embeddings_sum = torch.zeros_like(batch_text_embeddings[0][1])
    image_embeddings_sum = torch.zeros_like(batch_image_embeddings[0])

    for text_embeddings, image_embeddings in zip(batch_text_embeddings, batch_image_embeddings):
        head_embeddings_sum += text_embeddings[0]
        image_embeddings_sum += image_embeddings
        tail_embeddings_sum += text_embeddings[1]

    avg_expression = (head_embeddings_sum + image_embeddings_sum) / 2 - tail_embeddings_sum
    score = torch.norm(avg_expression, p=2).unsqueeze(0)

    head_prime_embeddings_sum = torch.zeros_like(batch_prime_text_embeddings[0][0])
    image_prime_embeddings_sum = torch.zeros_like(batch_prime_image_embeddings[0])

    for text_prime_embedings, image_prime_embeddings in zip(batch_prime_text_embeddings,batch_prime_image_embeddings):
        head_prime_embeddings_sum += text_prime_embedings[0]
        image_prime_embeddings_sum += image_prime_embeddings

    avg_expression_prime = (head_prime_embeddings_sum + image_prime_embeddings_sum) / 2 - tail_embeddings_sum
    score_prime = torch.norm(avg_expression_prime, p=2).unsqueeze(0)

    max_term = torch.max(torch.tensor([0]), score + CFG.gamma - score_prime)
    return max_term

def score_m_i(batch_image_embeddings,batch_motion_embeddings,batch_prime_image_embeddings):
    score = torch.tensor([0])
    score_prime = torch.tensor([0])

    length = len(batch_image_embeddings)
    length_prime = len(batch_prime_image_embeddings)

    if length > 1:
        length_prime = min(length_prime, length)
        for k in range(length-2):
            current_s = torch.norm(batch_image_embeddings[k] + batch_motion_embeddings[k] - batch_image_embeddings[k+1],
                                   p=2).unsqueeze(0)
            score += torch.max(torch.tensor([0]), current_s)
        if length_prime > 1:
            for m in range(length_prime-2):
                current_s_prime = torch.norm(batch_prime_image_embeddings[m] + batch_motion_embeddings[m] -
                                             batch_prime_image_embeddings[m+1], p=2).unsqueeze(0)
                score_prime += torch.max(torch.tensor([0]), current_s_prime)
        score_total = torch.max(torch.tensor([0]), score + CFG.gamma - score_prime)
    else:
        score_total = torch.max(torch.tensor([0]), score + CFG.gamma - score_prime)
    return score_total
