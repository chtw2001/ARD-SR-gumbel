# ARD-SR
The code implementation for ARD-SR: "Model-Agnostic Social Network Refinement with Diffusion Models for Robust Social Recommendation".


# Introduction
In this work, we propose a model-agnostic diffusion-based social network refinement framework for robust social recommendation (SR). It can be integrated seamlessly with any SR backbone for robust SRs.

# Requirements

The code has been tested running under Python 3.6.9. The required packages are as follows:
- torch ==1.9.0 --> A6000에서 지원하지 않아서 python 3.6.9에서 다운 가능한 최신 버전인 1.10.2+cu113 설치
- numpy == 1.19.5
- pandas==1.1.5
- `pip install scipy`

# An example to run ARD-SR  
 run ARD-SR on Ciao with MHCN as the SR backbone:

`python main.py --dataset Ciao`
