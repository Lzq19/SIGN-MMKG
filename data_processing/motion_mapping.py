import os
import shutil


def copy_frames(src_folder, dst_folder, frames_list):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    for frame in frames_list:
        src_path = os.path.join(src_folder, frame)
        dst_path = os.path.join(dst_folder, frame)
        shutil.copy(src_path, dst_path)


def write_to_txt(motion_trajectory_path,result_list):
    write_path = os.path.join(motion_trajectory_path, 'frame_record.txt')
    with open(write_path, 'w', encoding='utf-8') as f:
        for index, frame_seq in enumerate(result_list):
            f.write(f"{index}\t{' '.join(frame_seq)}\n")
    return write_path


# 定义输入和输出目录
img_dir = 'main/img'
all_frames_dir = 'main/all_frames'
motion_trajectory_dir = 'main/motion_trajectories'

# 获取img目录下的所有子目录（每个子目录代表一个场景或视频的关键帧文件夹）
video_folders = sorted(os.listdir(img_dir))

# 遍历每个场景或视频的关键帧文件夹
for video_folder in video_folders:
    result_list = []
    img_video_path = os.path.join(img_dir, video_folder)
    all_frames_video_path = os.path.join(all_frames_dir, video_folder)
    motion_trajectory_path = os.path.join(motion_trajectory_dir, video_folder)
    
    if  not os.path.exists(motion_trajectory_path):#如果路径不存在
        os.makedirs(motion_trajectory_path)

    # 获取当前场景文件夹中的所有关键帧文件名，并按文件名排序
    keyframes = sorted(os.listdir(img_video_path))
    
    # 如果当前文件夹中没有关键帧，直接跳过
    if not keyframes or len(keyframes)==1:
        result_list.append([])  # 添加空列表表示没有关键帧的情况
        write_to_txt(motion_trajectory_path,result_list)
        continue
    
    # 对于每两个相邻的关键帧，构建结果列表
    for i in range(len(keyframes) - 1):
        keyframe1 = keyframes[i]
        keyframe2 = keyframes[i + 1]
        

        all_frames = sorted(os.listdir(all_frames_video_path))
        
        # 获取两个关键帧文件夹中的所有帧文件名
        start_index = all_frames.index(keyframe1)
        end_index = all_frames.index(keyframe2)
        
        frame_sequence = all_frames[start_index:end_index + 1]
        
        # 构建当前关键帧对应的结果列表
        result_list.append(frame_sequence)

    write_path = write_to_txt(motion_trajectory_path,result_list)

    with open(write_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_parts = line.strip().split('\t')
            if len(line_parts) < 2:
                continue
            folder_index = line_parts[0]
            frames_list = line_parts[1].split()
            
            # 创建当前行对应的文件夹
            folder_path = os.path.join(motion_trajectory_path, folder_index)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 复制关键帧之间的所有帧到文件夹中
            copy_frames(all_frames_video_path, folder_path, frames_list)

