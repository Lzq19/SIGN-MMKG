import os
from typing import List
import torch
import json
import cv2
from config import CFG
from transformers import DistilBertTokenizer
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

def create_word_groups(text_path,tokenizer):

    def word_group_split(word_group,tokenizer):
        split_result = []
        current_result = []
        last_value = None
        for idx, little_group in enumerate(word_group):
            _, _, third_element = little_group
            relation_attribute = third_element.split("/")
            last_part = relation_attribute[-1]
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

        if current_result:
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
                    _, value, _ = parts

                    if value != current_word:
                        if current_group:
                            word_groups.append(current_group)
                        current_group = []
                        current_word = value

                    current_group.append(parts)
        
        if current_group:
            word_groups.append(current_group)

    for idx, word_group in enumerate(word_groups):
        word_groups[idx] = word_group_split(word_group,tokenizer)

    return word_groups


def load_imgs(image_path):

    def resize_and_normalize_image(image, size):
        resized_image = cv2.resize(image, size)
        normalized_image = resized_image.astype(np.float32) / 255.0
        return normalized_image

    all_images = []
    tmp_pth = image_path
    folder_names = sorted(os.listdir(image_path))

    for folder_name in folder_names:
        image_path = tmp_pth
        folder_path = os.path.join(image_path, folder_name)
        
        folder_images = []

        image_files = sorted(os.listdir(folder_path))
        
        for img_file in image_files:
            if img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.jpeg'):
                image_path = os.path.join(folder_path, img_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                target_size = (CFG.size, CFG.size)
                image = resize_and_normalize_image(image, target_size)
                image = torch.tensor(image).permute(2, 0, 1).float()
                if image is not None:
                    folder_images.append(image)
                else:
                    print(f"Could not read image: {img_file} in folder {folder_name}")
        
        all_images.append(folder_images)
    
    return all_images


def create_motion_trajectory(motion_path):

    def compute_trajectories(data_path, trajectories: List[str]) -> torch.Tensor:

        def remove_redundant_element(data):
            new_data = [item for index, item in enumerate(data) if (index + 1) % 3 != 0]
            return new_data

        all_coordinates = []
        pose_keypoints_size = 75
        face_keypoints_size = 210
        hand_keypoints_size = 63
        
        for file_name in trajectories:
            file_path = os.path.join(data_path, file_name)
            with open(file_path) as f:
                data = json.load(f)['people']
                if data:
                    data = data[0]
                    pose = data['pose_keypoints_2d']
                    face = data['face_keypoints_2d']
                    left_hand = data['hand_left_keypoints_2d']
                    right_hand = data['hand_right_keypoints_2d']
                else:
                    pose = [0] * pose_keypoints_size
                    face = [0] * face_keypoints_size
                    left_hand = [0] * hand_keypoints_size
                    right_hand = [0] * hand_keypoints_size
                combined_vector = pose + face + left_hand + right_hand
                all_coordinates.append(remove_redundant_element(combined_vector))
        
        coordinate_diff = []
        for i in range(1, len(all_coordinates)):
            diff = torch.tensor(all_coordinates[i]) - torch.tensor(all_coordinates[i-1])
            coordinate_diff.append(diff)
        
        trajectories_vector = torch.stack(coordinate_diff)
        
        return trajectories_vector
    
    motion_trajectory = []
    motion_folders = sorted(os.listdir(motion_path)) 

    for motion_folder in motion_folders: 
        word_trajectories = []
        betweens = [item for item in sorted(os.listdir(os.path.join(motion_path, motion_folder))) if ".txt" not in item]

        for between in betweens:
            data_path = os.path.join(motion_path, motion_folder, between)
            trajectories = sorted([item for item in os.listdir(data_path) if item.endswith('.json')])
            current_trajectory = compute_trajectories(data_path, trajectories)
            word_trajectories.append(current_trajectory)
        
        motion_trajectory.append(word_trajectories)

    def adjust_tensors_to_max_size(motion_trajectory):
        max_size = 0
        for sublist in motion_trajectory:
            for tensor in sublist:
                size = tensor.size(0)
                max_size = max(max_size, size)

        for sublist in motion_trajectory:
            for idx, tensor in enumerate(sublist):
                current_size = tensor.size(0)
                if current_size < max_size:
                    pad_size = max_size - current_size
                    padding = torch.zeros((pad_size,274))
                    tensor = torch.cat((tensor, padding), dim=0)
                    sublist[idx] = tensor

        print(f"motion_tensor_size: {max_size, 274}")
        return motion_trajectory

    motion_trajectory = adjust_tensors_to_max_size(motion_trajectory)

    return motion_trajectory


def Load_dataset(text_path, image_path, motion_path,tokenizer):
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
