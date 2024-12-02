# Nvidia-optical-flow-sample
A relatively simpler example of Nvidia optical flow API than the SDK provided by Nvidia

## Dependencies

1. FFMPEG (For getting the rawvideo frames of any video)
2. OpenCV (For displaying the output)
3. `NvOFInterface/` folder contains the header files which tells the names of the functions in API and their parameters and return types. It is essentially a more descriptive documentation than the one provided by Nvidia on their website. The latest updated files can be downloaded from [here](https://developer.nvidia.com/optical-flow-sdk).

## Usage

I have followed essentially the [API programming guide](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-programming-guide/index.html) provided by Nvidia. I highly recommend reading through and following it properly since I have implemented exactly as they have stated. This also contains quite a bit of the SDK code.

- First compile the code using `make` command.
- Then enter `./ofvec <path_to_the_video> <GPU_number> <Grid_size>`. 
- This should then pop up an OpenCV window showing the visualization of the flow vectors for the given video.

Feel free to modify, experiment with and use this code. I hope it serves as a basic starting point for people that are confused by the optical flow SDK and just want to calculate vectors between two frames.
