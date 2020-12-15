# INF573-Comic-Face

This project is aimed to replace a human face by a comic face by deeping learning method in a photo as well as in a video. We can also use it to replace your face by a comic face in a Zoom chat by simulatinga virtual camera, which is an interesting extension of this project.

### Setup

This project is built on several external dependencies.

The basic third part parckage are
* numpy
* opencv

Besides, we need to install and build environment for following 2 packages:
* [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) : to simulate a virtual camera by python.
* [dlib](http://dlib.net/): to introduce the face detection deep leaning model, which is inluded in model/shape_predictor_68_face_landmarks.dat in this project

### Main apis

The most simple way to use this project is by console lien as written in comic_camera.py, comic_video.py and comic_photo.py. This three files are used repsctively for :

* comic_camera.py: replace the real time camera in your computer, which can be used in Zoom
* comic_video.py: replace your face by any given comic face in a given video
* comic_photo.py: replace your face by any given comic face in a given photo

As those 3 APIs have most the same usage, we can take comic_photo.py as example.

First, to know how to use this file, enter the project path and type following code in youre console line:

`
python comic_photo.py
`

And we will get:

`
Usage: comic_photo.py comic_face.png person_photo.png [-m]

Option: -m : merge the comic face
`

Precisely, if we do not use "-m" which is optional, then we will simply replace the given human face by the given comic head (the given comic picture must contains only a head, like comic_pics/ki.png).
If we use "-m", then the given comic picture can be any front face of a comic figure with any backgroud (like comic_pics/sensei.png).
As an exemple, let us run:

`
python comic_photo.py comic_pics/ki.png human_pics/img.PNG

python comic_photo.py comic_pics/ki.png human_pics/img.PNG -m
`

The two output images are like:

![res1](results/replace.png "replacement result")![res2](results/merge.png "merged result")

If you want to use more flexible apis, we are welcomed to search for more usage in the codes!


### Other examples

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



### Acknowledge

During the development of this project, we are inspired by some existing projects and some codes have helped us to finish it.

They are namely:

* [mtcnn](https://github.com/ipazc/mtcnn)
* [dlib](https://github.com/davisking/dlib)
* [face morpher](https://github.com/alyssaq/face_morpher)
