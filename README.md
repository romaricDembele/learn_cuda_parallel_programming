# Cuda Parallel Programming Learning

## Environment Setup

### 1. Install WSL on Windows with Ubuntu distribution

### 2. Install Visual Studio on Windows
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

### 4. Uninstall CUDA Toolkit from WSL
```bash
# Run the binary file
/usr/local/cuda-12.6/bin/cuda-uninstaller
```



## Resources
- [TOP500 Supercomputer List](https://www.top500.org/lists/top500/)
- [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs)
- [NVIDIA Corporate Timeline](https://www.nvidia.com/en-us/about-nvidia/corporate-timeline/)
