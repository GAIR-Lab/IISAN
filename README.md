# [SIGIR2024] IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
![PEFT](https://img.shields.io/badge/Task-PEFT-red) 
![Recommendation](https://img.shields.io/badge/Task-Recommendation-red) 
<a href="https://arxiv.org/abs/2404.02059" alt="arXiv"><img src="https://img.shields.io/badge/arXiv-2404.02059-FAA41F.svg?style=flat" /></a>
<a href="https://mp.weixin.qq.com/s/fiCiPehSmDCM8ys3-xMwQg" alt="中文博客"><img src="https://img.shields.io/badge/博客-中文-orange.svg?style=flat" /></a> 
<a href="https://zhuanlan.zhihu.com/p/696297979" alt="知乎"><img src="https://img.shields.io/badge/知乎-中文-%23002FA7.svg?style=flat" /></a> 

# [TKDE] Efficient and effective adaptation of multimodal foundation models in sequential recommendation
<a href="https://arxiv.org/abs/2411.02992" alt="arXiv"><img src="https://img.shields.io/badge/arXiv-2411.02992-FAA41F.svg?style=flat" /></a>

If you are interested in adopting parameter-efficient fine-tuning (PEFT) in recommendation you can also refer to our previous WSDM 2024 paper: 
[Adapter4Rec](https://github.com/westlake-repl/Adapter4Rec)

### TODO list sorted by priority
* [x] Release the IISAN(Uncached)
* [x] Release baseline approaches
* [x] Release the IISAN(Cached)
--By April 30, 2024 (Completed early on April 15, 2024)
* [x] Release Datasets and IISAN(Cached)'s hidden states
* [] Release the Implementation of IISAN-Versa --By October 15th, 2025

      

**If you encounter any questions or discover a bug within the paper or code, please do not hesitate to open an issue or submit a pull request.**

## IISAN Introduction
Multimodal foundation models are transformative in sequential recommender systems, leveraging powerful representation learning capabilities. While Parameter-efficient Fine-tuning (PEFT) is commonly used to adapt foundation models for recommendation tasks, most research prioritizes parameter efficiency, often overlooking critical factors like GPU memory efficiency and training speed. Addressing this gap, our paper introduces  IISAN (Intra- and Inter-modal Side Adapted Network for Multimodal Representation), a simple plug-and-play architecture using a Decoupled PEFT structure and exploiting both intra- and inter-modal adaptation. 

IISAN matches the performance of full fine-tuning (FFT) and state-of-the-art PEFT. More importantly, it significantly reduces GPU memory usage — from 47GB to just 3GB for multimodal sequential recommendation tasks.  Additionally, it accelerates training time per epoch from 443s to 22s compared to FFT. This is also a notable improvement over the Adapter and LoRA, which require 37-39 GB GPU memory and 350-380 seconds per epoch for training. 

Furthermore, we propose a new composite efficiency metric, TPME (Training-time, Parameter, and GPU Memory Efficiency) to alleviate the prevalent misconception that "parameter efficiency represents overall efficiency". TPME provides more comprehensive insights into practical efficiency comparisons between different methods. Besides, we give an accessible efficiency analysis of all PEFT and FFT approaches, which demonstrate the superiority of IISAN.

![](figs/Framework.png) 


## IISAN-Versa Introduction
IISAN was limited to symmetrical MFMs and identical text and image encoders, preventing the use of state-of-the-art Large Language Models. To overcome this, we developed IISAN-Versa, a versatile plug-and-play architecture compatible with both symmetrical and asymmetrical MFMs. IISAN-Versa employs a Decoupled PEFT structure and utilizes both intra- and inter-modal adaptation. It effectively handles asymmetry through a simple yet effective combination of group layer-dropping and dimension transformation alignment. Our research demonstrates that IISAN-Versa effectively adapts large text encoders, and we further identify a scaling effect where larger encoders generally perform better. IISAN-Versa also demonstrates strong versatility in our defined multimodal scenarios, which include raw titles and captions generated from images and videos. Additionally, IISAN-Versa achieved state-of-the-art performance on the Microlens public benchmark. We will release our code and datasets to support future research.

![](figs/Framework-iisan-versa.png) 

## Experiment Setup
```
conda create -n iisan python=3.8

conda activate iisan

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 loralib==0.1.1 transformers==4.20.1 lmdb pandas
```
## Preparation
The complete textual recommendation datasets are available under the Dataset directory. 

Download the image files:

"am_image_is.zip" for Scientific dataset from this [link](https://drive.google.com/file/d/1vXLls-2DvvkMfgiCv3nB8C29nu0NDdt3/view?usp=sharing)

"am_image_mi.zip" for Instruments dataset from this [link](https://drive.google.com/file/d/1icKbl3ltN28WDESSKOdhOA0pNWIiNxe0/view?usp=sharing)

"am_image_op.zip" for Office dataset from this [link](https://drive.google.com/file/d/1zl-RbSNwUmQicCB3w1xY9RQWN8vGg5pC/view?usp=sharing)

You should unzip these zip files under "Dataset/". Then run the following to get the lmdb files:
```
cd Dataset/
python build_lmdb.py
```
Download "pytorch_model.bin" of the vit-base-patch16-224 from this [link](https://huggingface.co/google/vit-base-patch16-224) and bert-base-uncased from this [link](https://huggingface.co/google-bert/bert-base-uncased). Then put them under the respective subfolder under "pretrained_models/".

## Training & Testing for IISAN(Uncached)
```
cd Code_Uncached/scripts/
python run_IISAN.py
```
## Training & Testing for IISAN(Cached) 
**Note: Theoretically, IISAN(Cached) will only improve the training efficiency and maintain the original performance of IISAN(Uncached).**
```
cd Code_Cached/
python preprocess_vectors.py

cd scripts/
python run_IISAN.py
```

## Efficiency Analysis
![](figs/efficiency-analysis.png) 

## New Efficiency Metric - TPME (Training-time, Parameter, GPU Memory Efficiency)
<p align="center" width="100%">
<img src="figs/efficiency-metric.png" width="500"/>
</p>

where $\alpha$ denotes the weighting assigned to each term, tailored to specific circumstances, for example, in scenarios where only a limited GPU capacity is available for model training, it's advisable to significantly augment the weight of $M$. Within the scope of this paper, we've adjusted the values of $\alpha_1$ and $\alpha_3$ to 0.45, and $\alpha_2$ to 0.1. This adjustment reflects our focus on two key practical aspects: training speed and memory efficiency.

## Citation
If you find our paper useful in your work, please cite our paper as:
```
@inproceedings{fu2024iisan,
  title={IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT},
  author={Fu, Junchen and Ge, Xuri and Xin, Xin and Karatzoglou, Alexandros and Arapakis, Ioannis and Wang, Jie and Jose, Joemon M},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={687--697},
  year={2024}
}
@article{fu2024efficient,
  title={Efficient and effective adaptation of multimodal foundation models in sequential recommendation},
  author={Fu, Junchen and Ge, Xuri and Xin, Xin and Karatzoglou, Alexandros and Arapakis, Ioannis and Zheng, Kaiwen and Ni, Yongxin and Jose, Joemon M},
  journal={arXiv preprint arXiv:2411.02992},
  year={2024}
}
```




## Join the GAIR Lab at the University of Glasgow

Our [**GAIR Lab**](https://gair-lab.github.io/), specializing in **generative AI solutions for information retrieval tasks**, is actively seeking **highly motivated Ph.D. students** with a strong background in **artificial intelligence**.

If you're interested, please contact **Prof. Joemon Jose** at [joemon.jose@glasgow.ac.uk](mailto:joemon.jose@glasgow.ac.uk).


<p align="center" width="100%">
<img src="figs/logo.png" width="800"/>
</p>
