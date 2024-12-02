import os
import cv2
import numpy as np
from keras.models import load_model
from moviepy.editor import VideoFileClip
import csv

# Function to predict if a frame is violent or non-violent with confidence threshold
def predict_frame(model, frame, threshold=0.7):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = frame_resized / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)
    prediction = model.predict(frame_resized)[0]
    return prediction[1] > threshold, prediction[1]  # Returns True if violent with confidence

# Function to blur a frame
def blur_frame(frame):
    return cv2.GaussianBlur(frame, (51, 51), 0)

# Function to process the video, print predictions, blur violent frames, and log the results
def process_video(input_video_path, output_video_path, model_path, log_file_path, threshold=0.7, window_size=5):
    model = load_model(model_path)
    video_clip = VideoFileClip(input_video_path)
    
    # Open the log file to write timestamps of violent scenes
    with open(log_file_path, mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Frame Number", "Timestamp (s)", "Confidence"])  # CSV header
        
        # List to track recent frame predictions within the sliding window
        frame_predictions = []
        fps = video_clip.fps

        # Open a writer for the output video
        height, width = video_clip.size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        total_frames = 0
        detected_violent_frames = 0

        def process_frame(frame, frame_number):
            nonlocal total_frames, detected_violent_frames
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            is_violent, confidence = predict_frame(model, frame_bgr, threshold)
            
            # Track recent frame predictions within the window
            frame_predictions.append(is_violent)
            if len(frame_predictions) > window_size:
                frame_predictions.pop(0)

            # Blur only if a majority of frames in the window are classified as violent
            if sum(frame_predictions) > window_size // 2:
                print(f"Blurring a violent frame at frame {frame_number} with confidence {confidence}")
                frame_bgr = blur_frame(frame_bgr)
                detected_violent_frames += 1
                
                # Log the frame number, timestamp, and confidence
                timestamp = frame_number / fps
                log_writer.writerow([frame_number, timestamp, confidence])

            # Write the processed frame to the output video
            total_frames += 1
            out.write(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            return frame_bgr  # Return the processed frame

        # Process the video and save the result
        output_clip = video_clip.fl(lambda gf, t: process_frame(gf(t), int(t * fps)))
        output_clip.write_videofile(output_video_path, codec='libx264')

        # Release video writer
        out.release()
        video_clip.close()

        # Print summary
        print("\n=== Summary ===")
        print(f"Total Frames: {total_frames}")
        print(f"Detected Violent Frames: {detected_violent_frames}")
        print(f"Processed video saved to {output_video_path}")

# Main execution
if __name__ == "__main__":
    input_video_path = 'D:/Detection of Violent Scenes/input_videos/WhatsApp Video 2024-10-24 at 7.55.02 PM.mp4'
    output_video_path = 'D:/Detection of Violent Scenes/blurred_videos/cartoon_video_blurred.mp4'
    model_path = 'D:/Projects/Detection of Violent Scenes/models/violence_detection_model.h5'
    log_file_path = 'D:/Projects/Detection of Violent Scenes/violent_scenes_log.csv'

    process_video(input_video_path, output_video_path, model_path, log_file_path)
