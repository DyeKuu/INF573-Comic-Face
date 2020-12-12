# INF573-Comic-Face

This project is aimed to replace a human face by a comic face by deeping learning method in a photo as well as in a video. We can also use it to replace your face by a comic face in a Zoom chat by simulatinga virtual camera, which is an interesting extension of this project.

### Setup

This project is built on several external dependencies.

The basic third part parckage are 
* numpy
* opencv

Besides, we need to install and build environment for following 2 packages:
* [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) : to simulate a virtual camera by python.
* [dlib](http://dlib.net/): to introduce the face detection deep leaning model

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
python comic_photo.py comic_pics/ki.png 
python comic_photo.py comic_pics/ki.png -m
`

The two output images are like :
![res1](results/replace.png "replacement result")![res2](results/merge.png "merged result")

