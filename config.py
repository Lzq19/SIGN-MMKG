import torch

##########Config##########
class CFG:
    debug = False
    text_path='/home/lzq/Project/SIGN-MMKG/main/kg_T.txt'
    image_path='/home/lzq/Project/SIGN-MMKG/main/img'
    motion_path='/home/lzq/Project/SIGN-MMKG/main/motion'
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
    motion_embedding= 512
    max_length = 512

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1
    
    t_i = 0.8
    i_m = 0.2
    gamma = 1
