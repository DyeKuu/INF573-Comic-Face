# INF573-Comic-Face

### Pipeline

1. Detect humain face by using MTCNN (https://github.com/davidsandberg/facenet), (https://github.com/ipazc/mtcnn). Label the position of the eyes and the mouth etc.

2. Label comic faces manually on the eyes and the mouth, etc. Calculated the H matrix using these labelled point.

3. Replace humain face by comic faces by H matrix.

4. Write an android application.

(5. If we have time, we can try an algorithm to find the comic figutre who is most similar with the given photo. Or implement MTCNN algorithm by ourselves in Pytorch (Now they are all implemented by Tensorflow).)
