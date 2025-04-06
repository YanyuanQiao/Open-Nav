# Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs

> **International Conference on Robotics and Automation (ICRA) 2025**  
> **Authors:** Yanyuan Qiao, Wenqi Lyu, Hui Wang, Zixu Wang, Zerui Li, Yuan Zhang, Mingkui Tan, Qi Wu

## 🧠 Abstract

Vision-and-Language Navigation (VLN) tasks require an agent to follow textual instructions to navigate through 3D environments. Traditional approaches use supervised learning methods, relying heavily on domain-specific datasets to train VLN models. Recent methods try to utilize closedsource large language models (LLMs) like GPT-4 to solve VLN tasks in zero-shot manners, but face challenges related to expensive token costs and potential data breaches in realworld applications. In this work, we introduce Open-Nav, a novel study that explores open-source LLMs for zero-shot VLN in the continuous environment. Open-Nav employs a spatial-temporal chain-of-thought (CoT) reasoning approach to break down tasks into instruction comprehension, progress estimation, and decision-making. It enhances scene perceptions with fine-grained object and spatial knowledge to improve LLM’s reasoning in navigation. Our extensive experiments in both simulated and real-world environments demonstrate that Open-Nav achieves competitive performance compared to using closed-source LLMs.

## 📄 Project Website & Paper

- **Website**: [https://sites.google.com/view/opennav](https://sites.google.com/view/opennav)
- **ArXiv**: [https://arxiv.org/pdf/2409.18794](https://arxiv.org/pdf/2409.18794)


## ✅ Project Status

☑️ Release **OpenNav_R2R-CE_100** for quick and cost-effective testing in simulated environments.  
☑️ Full implementation of **Open-Nav** available for both training and inference

## ⚙️ Prerequisites
### Installation

We recommend using **Python 3.8** with a conda environment:

```bash
conda create -n opennav python=3.8
conda activate opennav
```

#### Install Habitat and Dependencies
This project builds upon [Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN). Please follow the steps below:

1. You could follow the [Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN) to install [`habitat-lab`](https://github.com/facebookresearch/habitat-lab) and [`habitat-sim`](https://github.com/facebookresearch/habitat-sim) by following the official Habitat installation guide.
2. We use Habitat [`v0.1.7`](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7) in our experiments, the same version used in [VLN-CE](https://github.com/jacobkrantz/VLN-CE) to ensure compatibility.
3. You may refer to **requirements.txt** or **environment.yml** in this repository for the exact package versions used.

ℹ️ Note: Our installation instructions are adapted from Discrete-Continuous-VLN.

### Dataset

**OpenNav_R2R-CE_100**: [Download Here](https://drive.google.com/file/d/1SfrPWqCIiivwduCYPMe-Za1wOt4eU6G9/view?usp=sharing)


Please place the downloaded files under: 

> data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/


### Scenes: Matterport3D

We use **Matterport3D (MP3D)** scene reconstructions in this project.

You can obtain the dataset by following the instructions on the [official Matterport3D project page](https://niessner.github.io/Matterport/). The download script `download_mp.py` is required to fetch the scenes.

To download the scenes:

> ⚠️ Requires **Python 2.7**.

```bash
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```
### Adapted MP3D Connectivity Graphs for Continuous Environments

To support **continuous navigation**, we adapt the [MP3D connectivity graphs](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity) originally defined for discrete environments.

These adapted graphs ensure that:
- All **nodes** are located in navigable open space.
- All **edges** are fully traversable by the agent using [VLN-CE configurations](https://github.com/jacobkrantz/VLN-CE/tree/master/habitat_extensions/config).

You can download the adapted connectivity graphs from the following link:

🔗 [Download Adapted Graphs (Google Drive)](https://drive.google.com/file/d/1FDJzwne0KgoHvLHyBRuXMIqLD_BW-UrM/view?usp=sharing)

Each file corresponds to a specific MP3D scene and contains:
- A list of node positions (some inherited from the original graph, others newly added).
- Edges connecting adjacent nodes to ensure full connectivity in continuous space.

Expected directory structure:
```
- data/
  - scene_datasets/
    - mp3d/
      - {scene_id}/
        - {scene_id}.glb
        - {scene_id}_semantic.ply
        - {scene_id}.house
        - {scene_id}.navmesh
```

### Trained Network Weights

We provide several pre-trained models to support waypoint prediction and visual encoding in the Open-Nav framework.

#### 📍 Candidate Waypoint Predictor

Path: 
> waypoint_prediction/checkpoints/check_val_best_avg_wayscore

- [RGB-D (FoV 90) weights used in our paper](https://drive.google.com/file/d/16Vk3ummmyLvpQr16TzBL-iwZNlrELOdk/view?usp=sharing)
- [Depth-only (FoV 90, R2R-CE)](https://drive.google.com/file/d/1goXbgLP2om9LsEQZ5XvB0UpGK4A5SGJC/view?usp=sharing)

These models are used to predict candidate waypoints in the environment from visual input.


#### 🧠 Visual Encoder (ResNet-50 for Depth)

Path:
> data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth

- Download link: [ResNet-50 pretrained on Gibson for DD-PPO](https://zenodo.org/record/6634113/files/gibson-2plus-resnet50.pth)

This ResNet-50 depth encoder is trained for PointGoal navigation on the Gibson dataset and used to extract visual features from depth images.

#### 🤖 External VLM Models

Some external models are required for Scene Perception:

- [**SpatialBot**](https://github.com/BAAI-DCAI/SpatialBot)
- [**RAM (Recognize Anything Model)**](https://github.com/xinyu1205/recognize-anything)

Please refer to their respective repositories for model download and setup instructions. These models are used to get spatial visual information to support the reasoning process of open-source LLMs.

Clone or place them under the root directory:

Path: 
> recognize_anything/

> SpatialBot3B/


## 🚀 Inference

To run inference with Open-Nav, use the provided script:

```bash
bash run_OpenNav.bash
```

### 🔧 Choosing the Language Model
You can specify which LLM to use via the --llm argument in the script. Supported options include:

	• gpt4o (default): Uses GPT-4o via OpenAI API
	• Qwen2, Llama3.1, Gemma, Phi3, etc.: Open-source LLMs (require local deployment)

⚠️ Open-source LLMs must be deployed separately and configured before use.


### 📐 Modifying Evaluation Episodes
To change the number of evaluation episodes, edit the following field in:
```
habitat_extensions/config/vlnce_task.yaml
```
Locate this section and modify EPISODES_TO_LOAD:

```yaml
DATASET:
  TYPE: VLN-CE-v1
  SPLIT: val_unseen
  DATA_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/OpenNav_R2R-CE_100_bertidx.json.gz
  SCENES_DIR: data/scene_datasets/
  EPISODES_TO_LOAD: 1  # Change this to run more episodes
```


## 🙏 Acknowledgements

We acknowledge that some parts of our code are adapted from existing open-source projects. Specifically, we reference the following repositories: **[DiscussNav](https://github.com/LYX0501/DiscussNav)**, **[Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN)**, **[SpatialBot](https://github.com/BAAI-DCAI/SpatialBot)**, **[RAM](https://github.com/xinyu1205/recognize-anything)**


## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{qiao2025opennav,
  author    = {Yanyuan Qiao and Wenqi Lyu and Hui Wang and Zixu Wang and Zerui Li and Yuan Zhang and Mingkui Tan and Qi Wu},
  title     = {Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}
```
