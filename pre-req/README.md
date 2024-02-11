# SCET DL Workshop Prerequisites
- install [Conda package manager](https://www.anaconda.com/download)
- next create conda env
```bash
    conda create -n scet_workshop python=3.11 anaconda
```
- activate the env 
```bash
    conda activate scet_workshop
```
- now install the appropriate pytorch based on GPU and CPU
    - install the cuda version **IF YOU HAVE NVIDIA GPU** 
    ```bash
    conda install pytorch torchvision  pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    - **ELSE** install the CPU version
     ```bash
     conda install pytorch torchvision cpuonly -c pytorch
     ```

MISC : 
- [try setting powershell profile on windows to access anaconda easily](https://gist.github.com/guimondmm/1a2e47d73a191429c7f8b0c12729dc59)

- install the official python and jupyter extensions for [VS CODE](https://code.visualstudio.com/download) or any other preferred code editor.