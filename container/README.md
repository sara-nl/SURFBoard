# SURFBoard Container Infrastructure

Folder structure:
* `patches/` contains a variety of patch files for tools installed in the container (Dali, TAU, Pyprof)
* `build-dali.docker` is a Dockerfile which produces a customized, NVTX-annotated version of Dali
* `build-surfboard.docker` is a Dockerfile describing the main profiling container 

## Building

There are two ways to build:
* When Docker is available: `make -f Makefile.docker` will build a Docker container then convert to Singularity. Docker containers are produced as a side-effect and can be utilized for profiling if convenient.
* Without Docker: `make -f Makefile.singularity` translates the dockerfiles to Singularity recipes, then builds with Singularity.

Both methods result in a Singularity image file: `surfboard.sif`
