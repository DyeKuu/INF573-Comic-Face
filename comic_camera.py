from image.video import VirtualCamera
import sys


if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print(
            f'Usage: {sys.argv[0]} comic_face.png [-m]\nOption: -m : merge the comic face')
        sys.exit(1)

    ifMerge = len(sys.argv) == 3

    v = VirtualCamera(comic_path=sys.argv[1])
    v.run(merge=ifMerge)
