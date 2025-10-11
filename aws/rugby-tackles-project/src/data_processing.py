import json
import cv2
import os
import shutil
import numpy as np
from pathlib import Path
import pandas as pd


def load_annotations(json_path):
    """Load annotations from exported JSON file."""
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def extract_video_clips(video_path, annotations, output_dir, clip_length=40):
    """
    Extract video clips based on annotations and save them.
    Modified to take last N frames from each tagged region.
    
    Args:
        video_path: Path to source video
        annotations: Loaded annotations dictionary
        output_dir: Directory to save clips
        clip_length: Number of frames per clip
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    clips_info = []
    for tag in annotations['tags']:
        action = tag['name']
        if not action:
            continue
        
        start_frame = tag['frameRange'][0]
        end_frame = tag['frameRange'][1]
        
        # Calculate the actual start frame for last N frames
        n_frames = end_frame - start_frame
        if n_frames < clip_length:
            actual_start = start_frame
        else:
            actual_start = end_frame - clip_length
        
        # Set video to actual start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
        
        # Read frames
        frames = []
        frames_to_read = min(clip_length, n_frames)
        for _ in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        # Only save if we have enough frames
        if len(frames) >= clip_length // 2:  # At least half the desired length
            # Pad with last frame if needed
            while len(frames) < clip_length:
                frames.append(frames[-1])
            
            # Take exactly clip_length frames
            frames = frames[:clip_length]
            
            # Create clip name and folder
            clip_name = f"{action}_{actual_start}.mp4"
            clip_folder = output_dir / action
            # clip_folder = Path(clip_folder)  # Ensure it's a Path object - this was necessary in the CoLab notebook
            clip_folder.mkdir(exist_ok=True, parents=True)
            clip_path = clip_folder / clip_name
            
            # Save clip
            if frames:
                out = cv2.VideoWriter(
                    str(clip_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frames[0].shape[1], frames[0].shape[0])
                )
                
                for frame in frames:
                    out.write(frame)
                out.release()
                
                clips_info.append({
                    'clip_path': str(clip_path),
                    'action': action,
                    'start_frame': actual_start,
                    'end_frame': actual_start + len(frames),
                    'n_frames': len(frames)
                })
            else:
                print(f"Warning: No frames extracted for clip {clip_name}")
        else:
            print(f"Warning: Not enough frames for clip {action} starting at {actual_start}")
    
    cap.release()
    return pd.DataFrame(clips_info)


def analyze_dataset(clips_df):
    """Analyze the dataset without TensorFlow dependencies."""
    print("=== Dataset Analysis ===")
    print(f"Total clips: {len(clips_df)}")
    print(f"Actions: {clips_df['action'].value_counts()}")
    print(f"Frame counts: {clips_df['n_frames'].describe()}")

    # Check file sizes and existence
    missing_files = []
    file_sizes = []

    for clip_path in clips_df['clip_path']:
        if os.path.exists(clip_path):
            size = os.path.getsize(clip_path) / (1024 * 1024)  # MB
            file_sizes.append(size)
        else:
            missing_files.append(clip_path)

    if missing_files:
        print(f"Warning: {len(missing_files)} missing files")

    print(f"File sizes (MB): mean={np.mean(file_sizes):.2f}, "
          f"min={np.min(file_sizes):.2f}, max={np.max(file_sizes):.2f}")

    return {
        'total_clips': len(clips_df),
        'actions': clips_df['action'].value_counts().to_dict(),
        'missing_files': missing_files,
        'avg_file_size_mb': np.mean(file_sizes) if file_sizes else 0
    }


def create_train_val_split(clips_df, val_split=0.2, random_state=42):
    """Create train/validation split."""
    np.random.seed(random_state)

    # Split by action to maintain class balance
    train_dfs = []
    val_dfs = []

    for action in clips_df['action'].unique():
        action_clips = clips_df[clips_df['action'] == action]
        n_val = int(len(action_clips) * val_split)

        # Shuffle and split
        shuffled = action_clips.sample(frac=1, random_state=random_state)
        val_clips = shuffled.iloc[:n_val]
        train_clips = shuffled.iloc[n_val:]

        train_dfs.append(train_clips)
        val_dfs.append(val_clips)

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)

    return train_df, val_df


def process_video_data(video_path, annotations_path, output_dir, clip_length=40):
    """Main processing function."""
    print(f"Processing video: {video_path}")
    print(f"Using annotations: {annotations_path}")

    # Load annotations
    annotations = load_annotations(annotations_path)
    print(f"Found {len(annotations['tags'])} annotations")

    # Extract clips
    clips_df = extract_video_clips(video_path, annotations, output_dir, clip_length)
    print(f"Extracted {len(clips_df)} clips")

    return clips_df


if __name__ == "__main__":

    clip_len = 40
    # Extract clips from all matches
    anno_list = os.listdir('data/raw/rugby_7s/dataset_2025_01_23/ann/')
    anno_dfs = []
    for anno in anno_list:
        anno_name = anno.split('.')[0]
        video_path = f'data/raw/rugby_7s/dataset_2025_01_23/video/{anno_name}.mp4'
        annotations_path = f'data/raw/rugby_7s/dataset_2025_01_23/ann/{anno_name}.json'
        output_dir = f'data/raw/rugby_7s/dataset_2025_01_23/processed/{anno_name}/'

        print(anno_name)
        if os.path.exists(video_path) and os.path.exists(annotations_path):
            clips_df = process_video_data(
                video_path, annotations_path, output_dir, clip_length=40
            )

        else:
            print("Test data not found. Please update paths in the script.")
            print(f"Looking for video: {video_path}")
            print(f"Looking for annotations: {annotations_path}")

        anno_dfs.append(clips_df)

    full_df = pd.concat(anno_dfs)

    # Save metadata
    full_df.to_csv(os.path.join(output_dir, 'all_clips_metadata.csv'), index=False)

    # Analyze dataset
    analysis = analyze_dataset(full_df)

    # Create train/val split
    train_df, val_df = create_train_val_split(full_df)
    train_df.to_csv(os.path.join(output_dir, 'train_clips.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_clips.csv'), index=False)

    print(f"Train set: {len(train_df)} clips")
    print(f"Validation set: {len(val_df)} clips")
    # training_df = pd.concat(anno_dfs)
    # validation_df = process_video_data(
    #     '/content/data/rugby_7s/dataset_2025_01_23/video/usd_loyola.mp4',
    #     '/content/data/rugby_7s/dataset_2025_01_23/ann/usd_loyola.json',
    #     '/content/data/rugby_7s/dataset_2025_01_23/dataset/usd_loyola/',
    #     clip_length=clip_len)

    # # Test with your existing data
    # video_path = 'data/raw/rugby_7s/dataset_2025_01_23/video/usd_loyola.mp4'
    # annotations_path = 'data/raw/rugby_7s/dataset_2025_01_23/ann/usd_loyola.json'
    # output_dir = 'data/raw/rugby_7s/dataset_2025_01_23/processed/usd_loyola/'

    # if os.path.exists(video_path) and os.path.exists(annotations_path):
    #     clips_df, analysis = process_video_data(
    #         video_path, annotations_path, output_dir, clip_length=40
    #     )

    #     # Save metadata
    #     clips_df.to_csv(os.path.join(output_dir, 'clips_metadata.csv'), index=False)

    #     # Create train/val split
    #     train_df, val_df = create_train_val_split(clips_df)
    #     train_df.to_csv(os.path.join(output_dir, 'train_clips.csv'), index=False)
    #     val_df.to_csv(os.path.join(output_dir, 'val_clips.csv'), index=False)

    #     print(f"Train set: {len(train_df)} clips")
    #     print(f"Validation set: {len(val_df)} clips")

    # else:
    #     print("Test data not found. Please update paths in the script.")
    #     print(f"Looking for video: {video_path}")
    #     print(f"Looking for annotations: {annotations_path}")