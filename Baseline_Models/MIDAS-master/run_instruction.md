# Running\_instruction:

## Setup

To run MIDAS, the following requirements must be met:

* **CMake** and **Ninja** must be installed and added to your `PATH`.
* A **C/C++ compiler** is required. In this example, we use the **x64 Native Tools Command Prompt for VS 2022**.

## Dataset

MIDAS requires three files in the dataset:

1. **Shape file** (meta): number of entries or structure information (`pathMeta`).
2. **Feature file** (data): the streaming-edge CSV (`pathData`).
3. **Label file**: the anomaly labels (`pathLabel`).

Inside `Demo.cpp`, modify the following variables to point to your custom files:

```cpp
std::string pathMeta  = "path/to/your/shape.meta";   // shape file
std::string pathData  = "path/to/your/features.csv"; // feature file
std::string pathLabel = "path/to/your/labels.csv";   // label file
```

## Run

1. Choose which MIDAS variant to run by editing `Demo.cpp`. Comment or uncomment the instantiation lines, for example:

   ```cpp
   // Original MIDAS (core)
   MIDAS::NormalCore midas(2, 1024);

   // Relational MIDAS (R)
   // MIDAS::RelationalCore midas(2, 1024);

   // Filtering MIDAS (F)
   // MIDAS::FilteringCore midas(2, 1024, 1e3f);
   ```

   * To run the **core** implementation, keep only the `NormalCore` line and comment out the others.

2. Build and run from the repository root:

   ```bat
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S . -B build\release
   cmake --build build\release --target Demo
   cd build\release
   .\Demo.exe
   ```
