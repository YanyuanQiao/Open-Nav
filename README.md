# Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs

> **International Conference on Robotics and Automation (ICRA) 2025**  
> **Authors:** Yanyuan Qiao, Wenqi Lyu, Hui Wang, Zixu Wang, Zerui Li, Yuan Zhang, Mingkui Tan, Qi Wu

## Abstract

Vision-and-Language Navigation (VLN) tasks require an agent to follow textual instructions to navigate through 3D environments. Traditional approaches use supervised learning methods, relying heavily on domain-specific datasets to train VLN models. Recent methods try to utilize closedsource large language models (LLMs) like GPT-4 to solve VLN tasks in zero-shot manners, but face challenges related to expensive token costs and potential data breaches in realworld applications. In this work, we introduce Open-Nav, a novel study that explores open-source LLMs for zero-shot VLN in the continuous environment. Open-Nav employs a spatial-temporal chain-of-thought (CoT) reasoning approach to break down tasks into instruction comprehension, progress estimation, and decision-making. It enhances scene perceptions with fine-grained object and spatial knowledge to improve LLM’s reasoning in navigation. Our extensive experiments in both simulated and real-world environments demonstrate that Open-Nav achieves competitive performance compared to using closed-source LLMs.

## Project Website & Paper

- 📄 **Website**: [https://sites.google.com/view/opennav](https://sites.google.com/view/opennav)
- 📄 **ArXiv**: [https://arxiv.org/pdf/2409.18794](https://arxiv.org/pdf/2409.18794)

## Dataset

We have made our dataset publicly available for easy access. You can download it from:

**OpenNav_R2R-CE_100**: [Download Here]([https://drive.google.com/file/d/1O78Ox0_fbiYRU-J38oVn8fU7XTNOqd6e/view?usp=drive_link](https://drive.google.com/file/d/1SfrPWqCIiivwduCYPMe-Za1wOt4eU6G9/view?usp=sharing))


## TODOs

☑️ Release **OpenNav_R2R-CE_100** for quick and cost-effective testing in simulated environments.  
⬜ Release **Full implementation** of Open-Nav.  


## Acknowledgements

We acknowledge that some parts of our code are adapted from existing open-source projects. Specifically, we reference the following repositories: **[DiscussNav](https://github.com/LYX0501/DiscussNav)**  **[SpatialBot](https://github.com/BAAI-DCAI/SpatialBot)**  **[RAM](https://github.com/xinyu1205/recognize-anything)**


## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{qiao2025opennav,
  author    = {Yanyuan Qiao and Wenqi Lyu and Hui Wang and Zixu Wang and Zerui Li and Yuan Zhang and Mingkui Tan and Qi Wu},
  title     = {Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}
```
