from moviepy.editor import VideoFileClip
from imagePipeline import *

if __name__ == '__main__':
    input_video_path = '/Users/chuanhl/Project/CarND-Advanced-Lane-Lines/harder_challenge_video.mp4'
    output_video_path = '/Users/chuanhl/Project/CarND-Advanced-Lane-Lines/harder_challenge_video_output.mp4'
    clip1 = VideoFileClip(input_video_path).subclip(0, 48)
    output_clip = clip1.fl_image(image_pipeline)
    output_clip.write_videofile(output_video_path, audio=False)
