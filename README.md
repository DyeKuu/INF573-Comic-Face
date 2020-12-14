# INF573-Comic-Face

### Pipeline

1. Detect humain face by using MTCNN (https://github.com/davidsandberg/facenet), (https://github.com/ipazc/mtcnn). Label the position of the eyes and the mouth etc.
2. Label comic faces manually on the eyes and the mouth, etc. Calculated the H matrix using these labelled point.

3. Replace humain face by comic faces by H matrix.

4. Write an android application.

(5. If we have time, we can try an algorithm to find the comic figutre who is most similar with the given photo. Or implement MTCNN algorithm by ourselves in Pytorch (Now they are all implemented by Tensorflow).)

### How to run

To replace human face in `person_photo.png` with `comic_face.png`.
Type the command in command line. [-m] is a optional parameter, if -m is provided, we soft merge.

```shell
python comic_photo.py comic_face.png person_photo.png [-m]
```

To replace human face in `person_video.mp4` with `comic_face.png`.
Type the command in command line. [-m] is a optional parameter, if -m is provided, we soft merge.

```shell
python comic_video.py comic_face.png person_video.mp4 [-m]
```

To branch it to a video call like a Zoom session.
Type the command in command line. [-m] is a optional parameter, if -m is provided, we soft merge.

```shell
python comic_camera.py comic_face.png [-m]
```

### Example

```shell
python comic_photo.py comic_pics/ki.png human_pics/img.PNG
```

![direct collage](./results/fusion.jpg)

```shell
python comic_photo.py comic_pics/ki.png human_pics/img.PNG -m
```

![direct collage](./results/alpha.png)
