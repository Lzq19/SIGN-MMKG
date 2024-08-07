import os
from typing import List
import torch
import json
import cv2
from config import CFG
from transformers import AutoTokenizer
import numpy as np
import copy


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

def create_word_groups(text_path,tokenizer):
    '''
    Function Description: 将每个手语词的所有三元组放在一个list中, words_group中, 每个元素都是一个以手语词为中心的sub-KG.
    text_path: 文本数据路径.
    words_group: 最终的大list, 包含所有小list.
    current_group: 当前处理的小list.
    current_word: 当前小list的第二列的值(手语词).
    '''
    def word_group_split(word_group,tokenizer):
        '''
        Function Description: 针对一个以手语词为中心的sub-KG, 根据r7(时间属性)对三元组分组(b, d).
        word_group: 以手语词为中心的(单个)sub-KG.
        split_result: split后的结果, 每个元素中的所有list, r7都是相同的.
        current_result: 当前正在处理的(b, d).
        '''
        split_result = []
        current_result = []  # 当前组的结果列表
        last_value = None # 上一个小小列表的最后一个值
        for idx, little_group in enumerate(word_group):
            _, _, third_element = little_group
            relation_attribute = third_element.split("/")
            last_part = relation_attribute[-1]  # 获取最后一个值
            if len(word_group[idx])==3: 
                word_group[idx][-1] = relation_attribute
            else:
                print(f'word_group: {word_group} have error number')

            flattened_list = [item for sublist in little_group for item in (sublist if isinstance(sublist, list) else [sublist])]
            encode_list = tokenizer(
                flattened_list, padding='max_length', truncation=False, max_length=CFG.max_length, return_tensors='pt'
                )
            if last_value is None or last_part == last_value:
                current_result.append(encode_list)
            else:
                split_result.append(current_result)
                current_result = [encode_list]

            last_value = last_part

        if current_result:  # 处理最后一个小列表
            split_result.append(current_result)

        return split_result
    
    word_groups = []
    current_group = []
    current_word = None

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 3:
                    _, value, _ = parts  # 获取第二列的值

                    if value != current_word:
                        if current_group:  # 如果当前小list不为空，则加入到大list中
                            word_groups.append(current_group)
                        current_group = []  # 开始一个新的小list
                        current_word = value  # 更新当前值

                    current_group.append(parts)  # 将行数据加入当前小list中

        if current_group:  # 处理最后一个小list
            word_groups.append(current_group)

    for idx, word_group in enumerate(word_groups):
        word_groups[idx] = word_group_split(word_group,tokenizer)

    return word_groups


def load_imgs(image_path):
    '''
    Function Description: 读取所有图片.
    image_path: 图片文件路径.
    all_images: 所有图片都在一个list中, 其中每个元素是一个手语词的图片list.(all_images是一个嵌套list)
    '''
    def resize_and_normalize_image(image, size):
        '''
        Function Description: resize和初始化.
        '''
        # Resize the image
        resized_image = cv2.resize(image, size)
        
        # Normalize the image
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        return normalized_image

    all_images = []
    tmp_pth = image_path
    # 获取指定路径下的所有文件夹列表
    folder_names = sorted(os.listdir(image_path))

    # 遍历每个文件夹，按顺序读取图片
    for folder_name in folder_names:
        image_path = tmp_pth
        folder_path = os.path.join(image_path, folder_name)
        
        # 用于存储当前文件夹内所有图片的列表
        folder_images = []
        
        # 获取当前文件夹内的所有图片文件，并按顺序读取
        image_files = sorted(os.listdir(folder_path))
        
        for img_file in image_files:
            if img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.jpeg'):
                image_path = os.path.join(folder_path, img_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                target_size = (CFG.size, CFG.size)
                image = resize_and_normalize_image(image, target_size)
                image = torch.tensor(image).permute(2, 0, 1).float()
                image
                if image is not None:
                    folder_images.append(image)
                else:
                    print(f"Could not read image: {img_file} in folder {folder_name}")
        
        # 将当前文件夹内的图片列表添加到总的图片列表中
        all_images.append(folder_images)
    
    return all_images


def create_motion_trajectory(motion_path):
    '''
    Function Description: 获取动作轨迹.
    motion_path: keypoints.json文件路径.
    motion_trajectory: 所有轨迹向量都在一个list中, 其中每个元素是一个手语词的轨迹list.(motion_trajectory是一个嵌套list)
    '''

    def compute_trajectories(data_path, trajectories: List[str]) -> torch.Tensor:
        '''
        Function Description: 计算两个json文件的运动轨迹.
        data_path: 当前json文件的根目录.
        trajectories: 当前的json文件.
        '''

        def remove_redundant_element(data):
            '''
            Function Description: 移除置信度.
            data: 当前json文件keypoints中所有值.
            '''
            new_data = [item for index, item in enumerate(data) if (index + 1) % 3 != 0]
            return new_data

        all_coordinates = []
        pose_keypoints_size = 75
        face_keypoints_size = 210
        hand_keypoints_size = 63
        # 读取每个.json文件，提取坐标信息并拼接成一个向量
        for file_name in trajectories:
            file_path = os.path.join(data_path, file_name)
            with open(file_path) as f:
                data = json.load(f)['people']
                if data:
                    data = data[0]
                    pose = data['pose_keypoints_2d']
                    # print('pose_size:', len(pose))
                    face = data['face_keypoints_2d']
                    # print('face_size:', len(face))
                    left_hand = data['hand_left_keypoints_2d']
                    # print('left_hand_size:', len(left_hand))
                    right_hand = data['hand_right_keypoints_2d']
                    # print('right_hand_size:', len(right_hand))
                else:
                    pose = [0] * pose_keypoints_size
                    face = [0] * face_keypoints_size
                    left_hand = [0] * hand_keypoints_size
                    right_hand = [0] * hand_keypoints_size
                # 将pose, face和hand拼接成一个向量
                combined_vector = pose + face + left_hand + right_hand
                # print('combined_vector_size:', len(combined_vector))
                all_coordinates.append(remove_redundant_element(combined_vector))
        
        # 计算坐标差并存储在张量中
        coordinate_diff = []
        for i in range(1, len(all_coordinates)):
            diff = torch.tensor(all_coordinates[i]) - torch.tensor(all_coordinates[i-1])
            # print('diff_size:', len(diff))
            coordinate_diff.append(diff)
        

        # torch.stack堆叠向量会增加一个维度，torch.cat不会增加维度
        trajectories_vector = torch.stack(coordinate_diff)
        # print('trajectories_vector.size:', trajectories_vector.size())
        # # dim=0结果是行向量，dim=1结果是列向量
        # trajectories_vector_t = torch.cat(coordinate_diff,dim=0)
        # print('trajectories_vector_t.size:', trajectories_vector_t.size())
        
        return trajectories_vector
    
    motion_trajectory = []
    motion_folders = sorted(os.listdir(motion_path)) 

    # motion_folder下是整个手语词的轨迹
    for motion_folder in motion_folders: 
        word_trajectories = []
        betweens = [item for item in sorted(os.listdir(os.path.join(motion_path, motion_folder))) if ".txt" not in item]

        # between是两个(b,d)之间的轨迹，如：“看”手语词有四个keyframes，motion_folder下就有三个group
        for between in betweens:
            data_path = os.path.join(motion_path, motion_folder, between)
            trajectories = sorted([item for item in os.listdir(data_path) if item.endswith('.json')])
            current_trajectory = compute_trajectories(data_path, trajectories)
            word_trajectories.append(current_trajectory)
        
        motion_trajectory.append(word_trajectories)

    def adjust_tensors_to_max_size(motion_trajectory):
        # 计算最大的张量大小
        max_size = 0
        for sublist in motion_trajectory:
            for tensor in sublist:
                size = tensor.size(0)
                max_size = max(max_size, size)

        # 调整张量大小并用零填充
        for sublist in motion_trajectory:
            for idx, tensor in enumerate(sublist):
                current_size = tensor.size(0)
                if current_size < max_size:
                    pad_size = max_size - current_size
                    padding = torch.zeros((pad_size,274))
                    tensor = torch.cat((tensor, padding), dim=0)
                    sublist[idx] = tensor

        print(f"张量大小已统一: {max_size, 274}")
        return motion_trajectory

    motion_trajectory = adjust_tensors_to_max_size(motion_trajectory)

    return motion_trajectory


def Load_dataset(text_path, image_path, motion_path,tokenizer):
    '''
    Function Description: 将一个手语词对应的文本, 关键帧, 动作轨迹放在一个list中, dataset包含了所有手语词对应的list.
    dataset = ['text':[], 'image':[], 'motion':[]][...]
    'text'中有m个list, 每个list都是一个group, 基本单位为9元组([body part, sign language word, relation(7个元素)]).
    '''
    dataset = []
    all_text_data = create_word_groups(text_path,tokenizer)
    all_imgs=load_imgs(image_path)
    motion_trajectory = create_motion_trajectory(motion_path)

    if len(all_text_data)==len(all_imgs)==len(motion_trajectory):
        for i in range(len(all_text_data)):
            if len(all_text_data[i])==len(all_imgs[i])==(len(motion_trajectory[i])+1):
                dataset.append({'text': all_text_data[i], 'image': all_imgs[i], 'motion': motion_trajectory[i]})
            else:
                print(f'Case {i} have error number')
    else:
        print('Error data number')
    return dataset


if __name__=='__main__':
    # Test
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    dataset = Load_dataset(CFG.text_path,CFG.image_path,CFG.motion_path,tokenizer)
    dataset = CustomDataset(dataset)
    print('Finish')
    # all_text_data = create_word_groups(text_path)
    # all_imgs=load_imgs(image_path)
    # motion_trajectory = create_motion_trajectory(motion_path)


