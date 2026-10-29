<p align="center">

  <h1 align="center">HUSH: Holistic Panoramic 3D Scene Understanding using Spherical Harmonics</h1>

  <p align="center">
    <a href="https://github.com/Syniez" rel="external nofollow noopener" target="_blank"><strong>Jongsung Lee</strong></a>
    ·
    <a href="https://github.com/Harin99" rel="external nofollow noopener" target="_blank"><strong>Harin Park</strong></a>
    ·
    <a href="https://sites.google.com/view/bulee" rel="external nofollow noopener" target="_blank"><strong>Byeong-Uk Lee</strong></a>
    ·
    <a href="https://vision3d-lab.github.io/" rel="external nofollow noopener" target="_blank"><strong>Kyungdon Joo</strong></a>
  </p>

<div align='center'>
  <a href='https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_HUSH_Holistic_Panoramic_3D_Scene_Understanding_using_Spherical_Harmonics_CVPR_2025_paper.pdf'><img src='https://img.shields.io/badge/Paper-CvF-blue'></a>
  <a href='https://vision3d-lab.github.io/hush/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
  <a href='https://github.com/vision3d-lab/HUSH'><img src='https://img.shields.io/badge/Video-E33122?logo=Youtube'></a>
</div>

<br>

This is official PyTorch implementation of **"HUSH: Holistic Panoramic 3D Scene Understanding using Spherical Harmonics"** (CVPR 2025)


## Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Train and Test](#train-and-test)
4. [Acknowledgements](#acknowledgements)
5. [BibTeX](#bibtex)
   
<br>

## Installation
The code is set up for a Linux cluster with **SLURM** and **`module load cuda`** (CUDA 12.x).
Everything else is installed with **conda** into the user's directory: **Python 3.11**, **PyTorch 2.9.1 (CUDA 12.6)**,
the remaining packages and the gcc used to build the Deformable Attention operator.
A step-by-step guide, including installing conda, other CUDA versions and troubleshooting, is in **[INSTALL.md](INSTALL.md)** (in Russian).

a. Download the source code:
```shell
git clone https://github.com/vision3d-lab/HUSH.git
cd HUSH
```

b. Create the conda environment:
```shell
conda env create -f environment.yml
```

c. The Deformable Attention operator ([idisc](https://github.com/SysCV/idisc/blob/main/docs/INSTALL.md)) is built automatically by the training job on its first run. To build it manually on a GPU node:
```shell
module load cuda
conda activate hush
bash models/ops/make.sh
```

<br>

## Dataset Preparation
Following the prior works, we used three benchmark datasets: **Stanford2D3D**, **Matterport3D**, and **Structured3D**.

a. **Stanford2D3D Dataset**

&nbsp;&nbsp;&nbsp;&nbsp;We follow the data organization noted at [Stanford2D3D](https://github.com/alexsax/2D-3D-Semantics).

b. **Matterport3D Dataset**

&nbsp;&nbsp;&nbsp;&nbsp;We used the processed stitched skybox Matterport3D dataset. <br>
>Please refer the [official repository](https://github.com/niessner/Matterport) and this [issue](https://github.com/niessner/Matterport/issues/13) for this step.  

c. **Structured3D Dataset**

&nbsp;&nbsp;&nbsp;&nbsp;We follow the data organization noted at [Structured3D](https://github.com/bertjiazheng/Structured3D/blob/master/data_organization.md).

d. **Layout Estimation (optional)**

&nbsp;&nbsp;&nbsp;&nbsp;For layout estimation, we need to pre-process the Matterport3D dataset to generate aligned panoramas. <br>
&nbsp;&nbsp;&nbsp;&nbsp;Official repository: [MatterportLayout](https://github.com/ericsujw/Matterport3DLayoutAnnotation?tab=readme-ov-file). <br>
>If you have problems during this process, this [issue](https://github.com/zhigangjiang/LGT-Net/issues/6) will be helpful. 

<br>

## Train and Test
- Train on SF2D3D with SLURM (adjust the `#SBATCH` lines first, see [INSTALL.md](INSTALL.md), section 6):
```shell
DATA_ROOT=/path/to/SF2D3D/ sbatch train_sf2d3d.sbatch
```
- The full training state is saved to `<ckpt_dir>/last.pth` after every epoch, and the best model to `<ckpt_dir>/best_model_SF2D3D.pth`. A new run continues from the last finished epoch, and the job requeues itself before the time limit.
- Train on the Matterport3D and Structured3D are also done similarly with train on the SF2D3D dataset.
- Test could be easily done by implementing validation functions and metric.py file.

<br>

## Acknowledgements
This work is built on several great research works, thanks a lot to all the authors for sharing their works.
- [AdaBins [CVPR 2021]](https://github.com/shariqfarooq123/AdaBins)
- [iDisc [CVPR 2023]](https://github.com/SysCV/idisc)
- [HRDFuse [CVPR 2023]](https://github.com/haoai-1997/HRDFuse)
- [torch-harmonics](https://github.com/NVIDIA/torch-harmonics)

<br>

## BibTeX
```bib
@inproceedings{lee2025hush,
  title={HUSH: Holistic Panoramic 3D Scene Understanding using Spherical Harmonics},
  author={Lee, Jongsung and Park, Harin and Lee, Byeong-Uk and Joo, Kyungdon},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={16599--16608},
  year={2025}
}
```
