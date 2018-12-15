from moviepy.editor import VideoFileClip
from imagePipeline import *
from Video import Video

if __name__ == '__main__':
    video = Video()
    input_video_path = '/Users/chuanhl/Project/CarND-Advanced-Lane-Lines/project_video.mp4'
    output_video_path = '/Users/chuanhl/Project/CarND-Advanced-Lane-Lines/project_video_output.mp4'
    clip1 = VideoFileClip(input_video_path).subclip(0, 50)
    output_clip = clip1.fl_image(video.process)
    output_clip.write_videofile(output_video_path, audio=False)
