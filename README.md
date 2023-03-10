# Heat Diffusion over a 2D Surface

This is an interactive heat simulation over a 2D surface written in C++ and using SFML for graphics/input, with the use of CUDA to make the computations on the GPU.

You can play around with your mouse adding heat or cold to the simulation while it runs in real-time. You can also add barriers so that the heat can't pass through a certain region and change the mouse's effect radius with the scroll wheel.

The main focus was to implement a heat diffusion equation but since the wave equation is so similiar we decided to implement it as well. So in the code you can simply choose which equation you want, using an explicit scheme. The file to change would be ```evolve.cu``` inside the ```computeCurrentCell``` function where you can choose one of three equations. We could add some UI for this but oh well.

## Authors

- [@Tiago França](https://github.com/TaigoFr)
- [@Miguel França](https://github.com/miguelfranca)

### Requirements
You will need to have a CUDA-Capable GPU and NVCC installed. 

The Makefile uses the the -lcudart flag.

## Installation
1 - Run `apt-get install g++` to install g++ 

2 - Download the SFML SDK from: https://www.sfml-dev.org/download/sfml/2.5.1/,
unpack it and copy the files to your preferred location

3 - Create the following environment variable:

```bash
  SFML_ROOT="<path-to-SFML-folder>"
```

4 - Update your library path with:

```bash
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-SFML-folder>/lib/
```

5 - Download/clone my SFML framework called GraphicsFramework from:

https://github.com/miguelfranca/SFML_GraphicsFramework.git

Place this GraphicsFramework repository on ```../``` relative to this repository.
You can change this on the Makefile if you want.


## Running

To run simply execute the following command on the base folder

```bash
  run
```

## Controls

```Left click + drag``` to add heat to the simulation

```Right click + drag``` to add cold to the simulation

```Mouse wheel``` to change the radius of impact of heat/cold

```Esc``` to exit the application and close the window


## Screenshots/GIFs

![](https://github.com/miguelfranca/2D_Heat_Diffusion/blob/main/screenshots-gifs/barrier.gif?raw=true)
![](https://github.com/miguelfranca/2D_Heat_Diffusion/blob/main/screenshots-gifs/diffusion.gif?raw=true)
![](https://github.com/miguelfranca/2D_Heat_Diffusion/blob/main/screenshots-gifs/screenshot1.png?raw=true)
![](https://github.com/miguelfranca/2D_Heat_Diffusion/blob/main/screenshots-gifs/screenshot2.png?raw=true)
