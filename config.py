import torch


class CFG:
    debug = False
    text_path = './data/kg_T_self_160.txt'
    image_path = './data/img'
    motion_path = './data/motion'
    batch_size = 1
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-3
    text_encoder_lr = 1e-3
    motion_encoder_lr = 1e-3
    weight_decay = 1e-3
    patience = 2
    factor = 0.5
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = 'distilbert-base-uncased'
    text_embedding = 768
    text_tokenizer = 'distilbert-base-uncased'
    motion_embedding = 512
    max_length = 512

    pretrained = True
    trainable = True
    temperature = 1.0

    size = 224

    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

    t_i = 0.8
    i_m = 0.2
    gamma = 1
    deepspeed = False