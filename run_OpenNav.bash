flag="--exp_name cont-cwp-opennav-ori
      --exp-config run_OpenNav.yaml
      --llm gpt-4o-2024-08-06
      --api_key 123456
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_unseen
      "
CUDA_VISIBLE_DEVICES=0 python run.py $flag