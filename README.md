# Cuda Parallel Programming Learning

**Note**: 
- The content under the folder **resources** is not mine. It is from **Hamdy Sultan**.
- All the remaining content is mine.

## Environment Setup

### 1. Install WSL on Windows with Ubuntu distribution

### 2. Install Visual Studio on Windows (Optional if using Visual Studio Code)
Check the 3 Workload components in the installation process:
- Desktop development with C++
- Windows Application Development
- Linux development with C++

### 3. Install CUDA Toolkit 12.6.3 in WSL

- Download and install
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
sudo sh cuda_12.6.3_560.35.05_linux.run
```
```bash
# Remove the installer after installation
sudo rm cuda_12.6.3_560.35.05_linux.run
```

- Update PATH
```bash
# Open the file
nano ~/.bashrc
```
```bash
# Add this line at the end of the file
export PATH="$PATH:/usr/local/cuda-12.6/bin"
```
```bash
# Apply the changes
source ~/.bashrc
```

- Add cuda to the dynamic linker configuration file
```bash
# Open the file
sudo nano /etc/ld.so.conf
```
```bash
# Edit the file by adding this line at the end of the file
/usr/local/cuda-12.6/lib64
```
```bash
# Update the shared library cache
sudo ldconfig
```

### 4. Check the installation
```bash
# Check nvcc version
nvcc --version
```
```bash
# Check the GPU access
nvdia-smi
```

### 5. (If necessary) Uninstall CUDA Toolkit from WSL
```bash
# Run the binary file
/usr/local/cuda-12.6/bin/cuda-uninstaller
```

### 6. Support for vs-code

#### Install the 2 extensions:
- "ms-vscode.cpptools-extension-pack"
- "nvidia.nsight-vscode-edition"

#### Create a launch file (**.vscode/launch.json**) with the content
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/[path_to_program]"
        },
        {
            "name": "CUDA C++: Attach",
            "type": "cuda-gdb",
            "request": "attach"
        }
    ]
}
```
Replace **[path_to_program]** in the json above with your source code path. For example if your program entry is src/main.cu, then replace **[path_to_program]** by **src/main**.

#### Add the CUDA include path to VS Code's IntelliSense
Add the following file (**.vscode/c_cpp_properties.json**) with the json content:
```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/cuda/include"
            ],
            "defines": [],
            "compilerPath": "/usr/local/cuda/bin/nvcc",
            "cStandard": "c17",
            "cppStandard": "gnu++17",
            "intelliSenseMode": "linux-clang-x64"
        }
    ],
    "version": 4
} 
```

### 7. Run the app
```bash
# You can directly use the Makefile target "start" to compile and run the app
make start
```

### 8. Profiling the app
You need to Enable GPU Performance Counters:
- Open the NVIDIA Control Panel in the Windows host.
- Enable Developer Settings under the Desktop tab in the Control Panal.
    (Control Panel > Desktop > Enable Developer Settings)
- Go to Manage GPU Performance Counters under the Developer section.
    (Developer > Manage GPU Performance Counters)
- Select "Allow access to the GPU performance counter to all users".
    

```bash
# CUDA Kernel profiler (ncu - NVIDIA Nsight Compute CLI)
make profile-kernel
```

## Resources
- [TOP500 Supercomputer List](https://www.top500.org/lists/top500/)
- [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs)
- [NVIDIA Corporate Timeline](https://www.nvidia.com/en-us/about-nvidia/corporate-timeline/)
