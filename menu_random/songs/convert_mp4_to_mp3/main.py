import os

from moviepy.editor import *


def convert_mp4_to_mp3(mp4_file, mp3_file):
    video = AudioFileClip(mp4_file)
    video.write_audiofile(mp3_file)


def convert_all_files(folder_dir=os.path.join(os.getcwd(), "menu_random", "songs")):
    """
    This will walk through all files in the current directory and convert them to MP3.
    """
    print(f'folder_dir: {folder_dir}')
    for root, dirs, files in os.walk(folder_dir):
        for file in files:
            if file.endswith(".mp4"):
                print(f'processing file: {file}')
                mp4_file = os.path.join(root, file)
                mp3_file = os.path.join(root, file.replace(".mp4", ".mp3"))
                convert_mp4_to_mp3(mp4_file, mp3_file)
                print(f"Converted {mp4_file} to {mp3_file} successfully!")


if __name__ == "__main__":
    convert_all_files(os.path.join(os.getcwd(), ".."))
