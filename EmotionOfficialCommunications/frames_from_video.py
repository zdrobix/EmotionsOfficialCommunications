# Funcție ce extrage frame-uri odată la 5 secunde dintr-un video
import os
def extract_frames_from_video(video_path, output_path, seconds=5):
    """
    Extracts frames from a video file at specified intervals and saves them as images.

    Parameters:
    video_path (str): The path to the input video file.
    output_path (str): The directory where the extracted frames will be saved.
    seconds (int): The interval in seconds at which to extract frames.
    """
    import cv2
    import os

    # Check if the input video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    # Create output directory if it doesn't exist
    output_path_video = os.path.join(output_path, os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(output_path_video, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * seconds) # Calculate the interval in frames

    count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break if no more frames are available

        if frame_count % interval == 0:
            frame_filename = os.path.join(output_path_video, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            count += 1

        frame_count += 1

    cap.release()

if __name__ == "__main__":
    video_path = './input/video'
    for video in os.listdir(video_path):
        video_file = os.path.join(video_path, video)
        output_path = './input/frames_from_videos'
        try:
            extract_frames_from_video(video_file, output_path)
            print(f"Frames extracted and saved to {output_path}")
        except Exception as e:
            print(f"An error occurred: {e}")