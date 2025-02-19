import json
import cv2
import os
import shutil
import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd


def load_annotations(json_path):
    """Load annotations from exported JSON file."""
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def extract_video_clips(video_path, annotations, output_dir, clip_length=64):
    """
    Extract video clips based on annotations and save them.

    Args:
        video_path: Path to source video
        annotations: Loaded annotations dictionary
        output_dir: Directory to save clips
        clip_length: Number of frames per clip
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Process each annotation
    clips_info = []
    for tag in annotations['tags']:
        action = tag['name']
        if not action:
            continue

        start_frame = tag['frameRange'][0]
        end_frame = tag['frameRange'][1]

        # Set video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Calculate number of frames in sequence
        n_frames = end_frame - start_frame

        # Read frames
        frames = []
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        # breakpoint()
        # Split into clips of specified length
        for i in range(0, len(frames), clip_length):
            clip = frames[i:i + clip_length]

            # Only save if clip is complete
            if len(clip) == clip_length:
                clip_name = f"{action}_{start_frame + i}.mp4"
                clip_folder = output_dir / action
                clip_folder = Path(clip_folder)
                clip_folder.mkdir(exist_ok=True, parents=True)
                clip_path = clip_folder / clip_name

                # Save clip
                out = cv2.VideoWriter(
                    str(clip_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frames[0].shape[1], frames[0].shape[0])
                )

                for frame in clip:
                    out.write(frame)
                out.release()

                clips_info.append({
                    'clip_path': str(clip_path),
                    'action': action,
                    'start_frame': start_frame + i,
                    'end_frame': start_frame + i + clip_length
                })

    cap.release()
    return pd.DataFrame(clips_info)



def create_tf_dataset(clips_df, frame_size=(224, 224)):
    """
    Create TensorFlow dataset from clips. Use for local TF datasets.
    
    Args:
        clips_df: DataFrame with clip information
        frame_size: Target frame size (height, width)
    """
    def load_video(path_tensor, label):
        # Convert tensor to string
        path = path_tensor.numpy().decode('utf-8')
        
        # Read video file
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Failed to open video: {path}")
            return np.zeros((1, *frame_size, 3), dtype=np.float32), label
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            frame = cv2.resize(frame, frame_size)
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if not frames:
            print(f"No frames read from: {path}")
            return np.zeros((1, *frame_size, 3), dtype=np.float32), label
            
        return np.array(frames, dtype=np.float32), label
    
    # Create label mapping
    labels = sorted(clips_df['action'].unique())
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Create dataset
    paths = clips_df['clip_path'].values
    labels = [label_to_index[label] for label in clips_df['action']]
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda x, y: tf.py_function(
            load_video,
            [x, y],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset, label_to_index


def test_dataset_loading(clips_df):
    """Test dataset creation with a single clip."""
    # Take first clip
    test_df = clips_df.iloc[[0]]
    clip_path = test_df['clip_path'].iloc[0]
    
    # Test direct video loading first
    print(f"\nTesting direct video loading for: {clip_path}")
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print("Failed to open video directly with OpenCV")
    else:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read first frame")
        else:
            print(f"Successfully read frame with shape: {frame.shape}")
    cap.release()
    
    # Now test through TensorFlow
    print("\nTesting through TensorFlow dataset:")
    dataset, label_mapping = create_tf_dataset(test_df)
    
    try:
        for frames, label in dataset.take(1):
            print(f"Frames shape: {frames.shape}")
            print(f"Label: {label.numpy()}")
    except Exception as e:
        print(f"Error during dataset iteration: {str(e)}")
    
    return dataset, label_mapping


# Example usage
def process_video_data(video_path, annotations_path, output_dir, clip_length):
    """Main processing function."""
    # Load annotations
    annotations = load_annotations(annotations_path)

    # Extract clips
    clips_df = extract_video_clips(video_path, annotations, output_dir, clip_length)

    return clips_df


if __name__ == "__main__":
    clips_df = process_video_data(
        'data/rugby_7s/dataset_2025_01_23/video/usd_loyola.mp4',
        'data/rugby_7s/dataset_2025_01_23/ann/usd_loyola.json',
        'data/rugby_7s/dataset_2025_01_23/dataset/usd_loyola/',
        clip_length=40)
    
    # test_dataset_loading(clips_df)
    # Create full dataset
    full_dataset, label_mapping = create_tf_dataset(clips_df)

    # Local dataset creation
    # Add batching and prefetching for training
    training_dataset = full_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

    # Verify dataset structure
    for frames, labels in training_dataset.take(1):
        print(f"Batch shape: {frames.shape}")
        print(f"Labels shape: {labels.shape}")
        print(labels)

