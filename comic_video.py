import sys
from image.video import Video

if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print(
            f'Usage: {sys.argv[0]} comic_face.png person_video.mp4 [-m]\nOption: -m : merge the comic face')
        sys.exit(1)

    ifMerge = len(sys.argv) == 4

    a = Video(video_path=sys.argv[2], comic_path=sys.argv[1], merge=ifMerge)
    a.process_video()

    print("result saved in result/output.avi")
