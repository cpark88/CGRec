# Cracking the Code of Negative Transfer: A Cooperative Game Theoretic Approach for Cross-Domain Sequential Recommendation (CIKM '23)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fcpark88%2FSyNCRec&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Overview
***
Code for our CIKM 2023 (<https://uobevents.eventsair.com/cikm2023/>) Paper "Cracking the Code of Negative Transfer: A Cooperative Game Theoretic Approach for Cross-Domain Sequential Recommendation." 
![model_arch](https://github.com/cpark88/SyNCRec/blob/main/syncrec_github_arch.png)
We referred to the source code of S3Rec (<https://github.com/RUCAIBox/CIKM2020-S3Rec/tree/master>).

## Abstract
***
This paper investigates Cross-Domain Sequential Recommendation (CDSR), a promising method that uses information from multiple domains (more than three) to generate accurate and diverse recommendations, and takes into account the sequential nature of user interactions. 
The effectiveness of these systems often depends on the complex interplay among the multiple domains. 
In this dynamic landscape, the problem of negative transfer arises, where heterogeneous knowledge between dissimilar domains leads to performance degradation due to differences in user preferences across these domains.
As a remedy, we propose a new CDSR framework that addresses the problem of negative transfer by assessing the extent of negative transfer from one domain to another and adaptively assigning low weight values to the corresponding prediction losses. 
To this end, the amount of negative transfer is estimated by measuring the marginal contribution of each domain to model performance based on a cooperative game theory.
In addition, a hierarchical contrastive learning approach that incorporates information from the sequence of coarse-level categories into that of fine-level categories (e.g., item level) when implementing contrastive learning was developed to mitigate negative transfer.
Despite the potentially low relevance between domains at the fine-level, there may be higher relevance at the category level due to its generalised and broader preferences.
We show that our model is superior to prior works in terms of model performance on two real-world datasets across ten different domains. 

## Environment Setting
***
```bash
pip install -r requirements.txt
```


## Dataset
***
To facilitate smooth testing, we have uploaded a small-sized temporary raw data in src/dataset/raw. By executing the command below, the datasets necessary for model training will be generated and saved. 
```python
python make_dataset.py --data_name='amazon' --strd_ym='202312'
```


## Train
***

```bash
bash train.sh
```


## Cite
If you use our codes for your research, cite our paper:

```
@inproceedings{park2023cracking,
  title={Cracking the Code of Negative Transfer: A Cooperative Game Theoretic Approach for Cross-Domain Sequential Recommendation},
  author={Park, Chung and Kim, Taesan and Choi, Taekyoon and Hong, Junui and Yu, Yelim and Cho, Mincheol and Lee, Kyunam and Ryu, Sungil and Yoon, Hyungjun and Choi, Minsung and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={2024--2033},
  year={2023}
}
```


[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=cpark88)](https://github.com/anuraghazra/github-readme-stats)
