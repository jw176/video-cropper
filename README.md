# :movie_camera: video-cropper
A simple python script to crop (not trim) a video

![](crop_demo.gif)

## Installation

Clone the github repository and then run the following command in the root directory to install the python dependencies.

```
pip install -r requirements.txt
```

## Usage

```
usage: crop.py [-h] [-o OUTPUT] [-f FRAME] [-c left right top bottom] input

positional arguments:
  input                 The input video file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        The output cropped video file name
  -f FRAME, --frame FRAME
                        The reference frame to show when cropping
  -c left right top bottom, --crop-values left right top bottom
                        The specific pixel values for the bounding rectangle to crop. Values relative to top left origin of the source image. E.g, 300 1500 150 800
```

## Libraries used

 - [OpenCV](https://opencv.org/) - For the cropping GUI and reading/cropping/writing video files
 - [tqdm](https://tqdm.github.io/) - For the progress bar

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Author
Jack Whelan

You can find me on [LinkedIn](https://www.linkedin.com/in/jack-whelan-1707491aa) 

## License
This project is licensed under the [MIT](LICENSE) License.
