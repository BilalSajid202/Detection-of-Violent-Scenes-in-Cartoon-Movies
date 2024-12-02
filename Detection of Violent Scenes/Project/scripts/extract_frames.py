import os
import cv2

def extract_frames(video_path, output_folder, video_label):
    """Extract frames from a video and save them to an output folder."""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as JPG file with the appropriate label
        frame_filename = os.path.join(output_folder, f"{video_label}_frame{count}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames from {video_path}.")

def process_videos(video_folder, output_folder, label):
    """Extract frames from all videos in a folder."""
    for video_file in os.listdir(video_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_folder, video_file)
            extract_frames(video_path, output_folder, label)

if __name__ == "__main__":
    # Use absolute paths
    violent_videos_folder = 'D:/Projects/Detection of Violent Scenes/dataset/violent/'
    non_violent_videos_folder = 'D:/Projects/Detection of Violent Scenes/dataset/non_violent/'

    # Paths to save extracted frames
    violent_output_folder = 'D:/Projects/Detection of Violent Scenes/extracted_frames/violent/'
    non_violent_output_folder = 'D:/Projects/Detection of Violent Scenes/extracted_frames/non_violent/'

    # Extract frames from violent videos
    process_videos(violent_videos_folder, violent_output_folder, 'violent')

    # Extract frames from non-violent videos
    process_videos(non_violent_videos_folder, non_violent_output_folder, 'non_violent')
