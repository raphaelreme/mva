# 3D computer vision (First Semester)

## Usage
Compilation with CMake/Make tool chain. Uses the C++ language.

```bash
$ cd <tp_name>
$ mkdir build
$ cd build
$ cmake [-DCMAKE_BUILD_TYPE=Release] ..  # The option will optimise the code and thus running speed
$ make
$ ./<MyProgram>
```

# panorama

See exercise.pdf in the associated folder. The goal is to merge to neighboring photos in a single photo. (make a panorama from 2 photos)

# fundamental

See exercise.pdf in the associated folder. The goal is to compute the fundamental matrix with RANSAC algorithm.

# seeds

See exercise.pdf in the associated folder. The goal is to compute a disparity map using a seeds algorithm.

# gcdispar

The goal is to compute a disparity map but using a graph cut algorithm.
