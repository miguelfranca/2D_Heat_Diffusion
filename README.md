# 2D_Heat_Diffusion

## Authors

- [@Tiago França](https://github.com/TaigoFr)
- [@Miguel França](https://github.com/miguelfranca)

### Requirements
You will need to have a CUDA-Capable GPU and NVCC installed. 

The Makefile uses the the -lcudart flag.

## Installation
1 - Download the SFML SDK from: https://www.sfml-dev.org/download/sfml/2.5.1/,
unpack it and copy the files to your preferred location

2 - Create the following environment variable:

```bash
  SFML_ROOT="<path-to-SFML-folder>"
```

3 - Update your library path with:

```bash
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-SFML-folder>/lib/
```

4 - Download/clone my SFML framework called GraphicsFramework from:

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