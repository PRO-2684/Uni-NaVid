# Uni-NaVid Dev Container

This devcontainer configuration provides a complete development environment for the Uni-NaVid project.

## Features

- **Base Image**: NVIDIA CUDA 11.8.0 with cuDNN 8 on Ubuntu 22.04
- **Python Environment**: Conda environment with Python 3.10
- **GPU Support**: Configured for 2 GPUs (adjustable in `devcontainer.json`)
- **Pre-installed Dependencies**: All project dependencies including PyTorch, DeepSpeed, and flash-attention
- **VS Code Extensions**: Python, Pylance, Black formatter, isort, Jupyter, and more
- **Shell**: Zsh with Oh My Zsh for enhanced terminal experience

## Quick Start

1. **Prerequisites**:
   - Docker with NVIDIA Container Toolkit installed
   - VS Code with the "Dev Containers" extension
   - NVIDIA GPU with appropriate drivers

2. **Open in Container**:
   - Open this project in VS Code
   - Press `F1` and select "Dev Containers: Reopen in Container"
   - Wait for the container to build and initialize

3. **Verify Installation**:
   ```bash
   # Check Python version
   python --version  # Should show Python 3.10.x
   
   # Check conda environment
   conda info --envs  # Should show 'uninavid' environment
   
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Configuration

### GPU Settings

To adjust the number of GPUs, edit the `runArgs` in `devcontainer.json`:

```json
"runArgs": [
    "--gpus",
    "all",  // Use all available GPUs, or specify a number like "2"
    "--shm-size=16gb",
    "--ipc=host"
]
```

### Shared Memory

The container is configured with 16GB of shared memory (`--shm-size=16gb`) for PyTorch DataLoader. Adjust if needed based on your system.

### User Permissions

The container runs as the `vscode` user (UID 1000, GID 1000) to match typical Linux user permissions. Adjust the build args in `devcontainer.json` if needed:

```json
"args": {
    "USERNAME": "vscode",
    "USER_UID": "1000",
    "USER_GID": "1000"
}
```

## Post-Create Setup

The container automatically runs the following commands after creation:
1. Upgrades pip
2. Installs the project in editable mode (`pip install -e .`)
3. Installs flash-attention

If you need to reinstall dependencies manually:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate uninavid
pip install --upgrade pip
pip install -e .
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

## Directory Structure

- `/workspace`: Your project root (mounted from host)
- `/opt/conda`: Conda installation
- `/opt/conda/envs/uninavid`: The uninavid conda environment

## VS Code Settings

The container comes with pre-configured VS Code settings:
- Python interpreter: `/opt/conda/envs/uninavid/bin/python`
- Formatter: Black
- Linter: Flake8
- Auto-format on save enabled
- Auto-organize imports on save

## Troubleshooting

### Container fails to build
- Ensure Docker has enough disk space
- Check that NVIDIA Container Toolkit is properly installed
- Verify GPU drivers are up to date

### GPU not detected
- Check `nvidia-smi` output in the container
- Ensure `--gpus` flag is properly set in `runArgs`
- Verify host system has NVIDIA drivers installed

### Out of memory errors
- Increase `--shm-size` in `devcontainer.json`
- Reduce batch size in your training scripts
- Monitor GPU memory with `nvidia-smi`

### Permission issues
- Adjust USER_UID and USER_GID to match your host user
- Rebuild the container after changes

## Additional Resources

- [Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Uni-NaVid Project](https://pku-epic.github.io/Uni-NaVid/)
