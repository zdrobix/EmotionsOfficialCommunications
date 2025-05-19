# functie care ia video dintr un path si ii extrage audio
# in format wav
import os
import subprocess


def extract_audio_from_video(video_path, output_path):
    """
    Extracts audio from a video file and saves it as a WAV file.

    Parameters:
    video_path (str): The path to the input video file.
    output_path (str): The path where the output WAV file will be saved.
    """
    # Check if the input video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    # Use ffmpeg to extract audio from the video
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # Audio codec
        '-ar', '44100',  # Audio sample rate
        '-ac', '2',  # Number of audio channels
        output_path
    ]

    subprocess.run(command, check=True)  # Run the command and check for errors


if __name__ == "__main__":
    # Example usage
    # Path to the input video file
    video_path = './input/video/1578318-hd_1920_1080_30fps.mp4'
    # Path to save the extracted audio
    output_path = 'output/1578318-hd_1920_1080_30fps.wav'

    try:
        extract_audio_from_video(video_path, output_path)
        print(f"Audio extracted and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
