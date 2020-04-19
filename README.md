# facecog

This is a `dlib` and `OpenCV` test project for real-time face recognition.
Unlike other existing projects, I have made this application simple to extend.
Right now it identifies Elon Musk in the webcam stream (which you can maybe only satisfy using a photo on your phone).
More persons can be added to [persons](persons) directory with **1 or more** photo to use for recognition.

To run, you must obviously have `python` installed.
I would recommend Anaconda and creating a virtual environment with it.
For enhanced performance, you should install [cuda](https://developer.nvidia.com/cuda-downloads), [cudNN](https://developer.nvidia.com/cudnn) and [Intel Math Kernel Library](https://software.intel.com/en-us/mkl) before installing `dlib`.


Then clone this project, install dependencies and run the program:
```shell script
git clone https://github.com/vaibhavpandeyvpz/facecog
pip install -r requirements.txt
python facecog.py
```

### LICENSE
See the [LICENSE](LICENSE) file.
