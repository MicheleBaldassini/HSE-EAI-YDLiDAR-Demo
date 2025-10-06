# YDLidar-SDK Setup Guide

This repository provides the setup instructions to build and run the **YDLidar-SDK** on Windows using **Visual Studio 2017**, **CMake**, **vcpkg**, and **SWIG**.

---

## Requirements

Before you begin, download and install the following tools:

* [vcpkg](https://github.com/Microsoft/vcpkg)
* [Visual Studio 2017](https://visualstudio.microsoft.com/it/vs/older-downloads/)
* [CMake 3.22.0 (Windows x86_64 binary distribution)](https://cmake.org/download/)
* [SWIG 4.0.2 (swigwin)](http://www.swig.org/download.html)

---

## Installation Steps

1. **Unzip the downloaded archives**:

   * Extract `vcpkg`, `CMake`, and `SWIG`.

2. **Move the directories**:

   * Place `vcpkg` in:

     ```
     C:\Users\Username
     ```
   * Place `CMake` and `SWIG` in:

     ```
     C:\Program Files\
     ```

3. **Add environment variables**:
   Add the following paths to your system `PATH`:

   ```
   C:\Program Files\cmake\bin
   C:\Program Files\swigwin-4.0.2
   ```

---

## Configure vcpkg

Open a terminal and run:

```powershell
cd [vcpkg directory]
.\bootstrap-vcpkg
.\vcpkg integrate install
```

---

## Build the SDK

Inside the `YDLidar-SDK` repository:

```powershell
cd YDLidar-SDK
mkdir build
cd build
```

Replace `[vcpkgroot]` with the path of your `vcpkg` installation.

* **Generate the 32-bit project**:

  ```powershell
  cmake .. "-DCMAKE_TOOLCHAIN_FILE=[vcpkgroot]\scripts\buildsystems\vcpkg.cmake"
  ```

* **Generate the 64-bit project**:

  ```powershell
  cmake .. -G "Visual Studio 15 2017 Win64" "-DCMAKE_TOOLCHAIN_FILE=[vcpkgroot]\scripts\buildsystems\vcpkg.cmake"
  ```

---

## Python Installation

From the `YDLidar-SDK` root:

```powershell
pip install .
```

---

## Run the System

To launch the LiDAR:

```bash
./start_LiDAR.sh
```

---

## Demo

![LiDAR Demo](./docs/lidar_demo.gif)

---

## Contact 

For more information, please contact:
\u2709 **[your.email@example.com](mailto:your.email@example.com)**

---
