source /home/lingsheng/jiangyuhua/miniconda/etc/profile.d/conda.sh

# 仅 cosmos-policy + LIBERO（不含 dreamzero）
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/root/.cache \
  -v /home/lingsheng/chennuo/cosmos-policy:/workspace \
  -v /home/lingsheng/LIBERO-Cosmos-Policy:/workspace/LIBERO-Cosmos-Policy \
  -v /data1/lingsheng:/workspace/data1 \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace \
  --entrypoint bash \
  openpi:installed

# 带 DreamZero 挂载的 Docker（cosmos-policy + dreamzero 同容器）
# 目录层级：宿主机 cosmos-policy -> 容器 /workspace；宿主机 dreamzero -> 容器 /workspace/dreamzero
# 每次新建容器后需在容器内重新构建 uv 环境（见下方「每次新建容器后」）
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/root/.cache \
  -v /home/lingsheng/chennuo/cosmos-policy:/workspace \
  -v /home/lingsheng/chennuo/dreamzero:/workspace/dreamzero \
  -v /data1/lingsheng:/workspace/data1 \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace \
  --entrypoint bash \
  openpi:installed


apt-get update

apt-get install -y build-essential cmake ninja-build pkg-config \
                   libegl1-mesa-dev libgl1-mesa-dev libx11-dev

# --- 每次新建容器后（在 /workspace 下执行）---
# 容器内 /workspace 即宿主机 cosmos-policy，.venv 位于 /workspace/.venv。
# 每次 docker run 新建容器后建议执行以下命令，确保 uv 环境一致（否则可能因镜像/缓存差异导致依赖不一致）：
#   cd /workspace && uv sync --extra cu128 --group libero --python 3.10
# 若 .venv 已存在且未改动，可跳过；若换镜像或首次使用，必须执行。

# --- .venv 是否支持 DreamZero ---
# 结论：/workspace/.venv（即 cosmos-policy/.venv）不能完整支持 DreamZero 的官方环境。
# 原因简述：
#   - Python：cosmos-policy 使用 3.10（uv --python 3.10），DreamZero 要求 ~=3.11,<3.13。
#   - PyTorch：cosmos 为 torch 2.7.0+cu128，DreamZero 为 torch==2.8.0 / torchvision==0.23.0。
#   - 其它冲突：peft (cosmos>=0.17.1 vs dreamzero==0.5.0)、transformers (4.57.1 vs 4.51.3)、
#     diffusers (>=0.35.2 vs 0.30.2)、albumentations (>=2.0.8 vs 1.4.18)、opencv (>=4.11 vs 4.8.0.74) 等。
# 若要在同一容器内同时跑 cosmos-policy 与 dreamzero，可选：
#   (1) 仅用当前 .venv 跑 cosmos-policy，用下方 uv pip install 补充部分 dreamzero 依赖（会有版本折中，不保证 dreamzero 全部功能）。
#   (2) 为 DreamZero 单独建 venv：cd /workspace/dreamzero && uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[dev]" 等（推荐需完整 dreamzero 时使用）。

# --- DreamZero 源码功能与运行方式 ---
# 功能概览：
#   1) 推理服务 (WebSocket)：socket_test_optimized_AR.py
#      - 多 GPU 分布式加载 DreamZero-DROID 等 checkpoint，提供 WebSocket 策略接口；
#      - 支持 DiT 缓存加速（--enable-dit-cache）；兼容 RoboArena 协议，可接仿真/真机客户端。
#   2) 测试客户端：test_client_AR.py
#      - 向本地推理服务发观测（debug_image/ 下视频帧或 --use-zero-images 零图），收动作并打日志；
#      - 用于验证服务是否正常、观测/动作格式是否正确。
#   3) 仿真评估：eval_utils/run_sim_eval.py
#      - 依赖 sim_evals + Isaac Lab，在 DROID 仿真场景中跑多轮 rollout，连接本地策略服务（host/port）；
#      - 需单独 clone sim_evals、下载环境资源，并先启动上述推理服务。
#   4) 依赖检查：check_deps.py（需 Python 3.11+，见下）
#   5) 安装后控制台命令：dreamzero-server 等价于运行 socket_test_optimized_AR:main
#
# 如何运行（容器内已 source /workspace/.venv/bin/activate，且已挂载 /workspace/dreamzero）：
#   A. 下载 checkpoint（若尚未下载）：
#      export HF_ENDPOINT="https://hf-mirror.com"  # 可选
#      uvx hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir /workspace/data1/DreamZero-DROID
#   A'. 为何还要下载？--model-path 只提供「DreamZero 策略权重」和 config.json。config 里还引用了「基座」Wan2.1-I2V（T5、CLIP、VAE、DiT 等）；若这些路径在本地不存在，代码会从 HuggingFace 拉取。容器无外网时会报 Network is unreachable。解决办法：在宿主机有网时下载 Wan2.1 到本地，挂载进容器并设 WAN21_LOCAL_DIR：
#      宿主机：uvx hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /data1/lingsheng/Wan2.1-I2V-14B-480P
#      启动服务前在容器内：export WAN21_LOCAL_DIR=/workspace/data1/Wan2.1-I2V-14B-480P（并确保该目录已挂载且包含 models_t5_umt5-xxl-enc-bf16.pth、models_clip_*.pth、Wan2.1_VAE.pth、diffusion_pytorch_model.safetensors* 等）
#      若报错 Repo id must be in the form 'namespace/repo_name': '/mnt/aws-lfs-02/.../umt5-xxl'，说明 transform 的 tokenizer 路径指向训练机本地，需在宿主机下载 UMT5-XXL tokenizer 到基座目录下：hf download google/umt5-xxl --local-dir /data1/lingsheng/Wan2.1-I2V-14B-480P/umt5-xxl（容器内即 /workspace/data1/Wan2.1-I2V-14B-480P/umt5-xxl）
#   B. 启动推理服务（至少 2 张 GPU；若已按 A' 准备好基座，先 export WAN21_LOCAL_DIR=...；整条一行复制）：
#      cd /workspace/dreamzero && export PYTHONPATH=/workspace/dreamzero:$PYTHONPATH
#      CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path /workspace/data1/DreamZero-DROID
#   C. 运行测试客户端（必须与推理服务在同一个容器内，否则会 Connection refused）：
#      方式一：在宿主机另开一个终端，先查正在跑服务的容器 ID：docker ps（找到 openpi 那个），再进入同一容器：docker exec -it <容器ID> bash，然后 source /workspace/.venv/bin/activate，再执行：
#      cd /workspace/dreamzero && PYTHONPATH=/workspace/dreamzero python test_client_AR.py --host localhost --port 5000
#      若没有 debug_image/ 下的视频，可加 --use-zero-images 用零图测试。
#      方式二：若用 tmux/screen，在同一个容器里开两个 pane，一个跑 B 的服务，一个跑本命令。
#   D. 仿真评估（可选）：需先安装 sim_evals 与 Isaac Lab，并启动上述服务（端口与 run_sim_eval 的 --port 一致），再：
#      python eval_utils/run_sim_eval.py --host localhost --port 6000 --episodes 10 --headless
#      （注意 run_sim_eval 默认连 6000，若服务在 5000 需传 --port 5000）

# --- 检验当前环境能否运行 DreamZero（已 source /workspace/.venv/bin/activate）---
# 1) 最小导入测试（在容器内、已激活 .venv 时执行，工作目录与 PYTHONPATH 指向 dreamzero 根）
cd /workspace/dreamzero
export PYTHONPATH=/workspace/dreamzero:$PYTHONPATH
python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
from einops import rearrange
print('einops OK')
from openpi_client import base_policy
print('openpi_client OK')
try:
    from groot.vla.data.schema import EmbodimentTag
    print('groot.vla.data.schema OK')
except Exception as e:
    print('groot import failed:', e)
try:
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    print('GrootSimPolicy OK')
except Exception as e:
    print('GrootSimPolicy import failed:', e)
print('dreamzero minimal import test done.')
"
# 若上面有报错，多为依赖版本冲突（见上方 .venv 兼容性说明）；全部 OK 则可继续做 2）。
# 2) 依赖版本检查（需 Python 3.11+，因 check_deps.py 使用 tomllib；当前 .venv 为 3.10 则跳过或换用 dreamzero 独立 venv）
# python /workspace/dreamzero/check_deps.py --pyproject /workspace/dreamzero/pyproject.toml --venv /workspace/.venv
# 3) 实际运行推理服务（需已下载 checkpoint，例如 /workspace/data1/DreamZero-DROID）。整条一行复制：

export WAN21_LOCAL_DIR=/workspace/data1/Wan2.1-I2V-14B-480P

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path /workspace/data1/DreamZero-DROID
# 另开终端启动测试客户端：
# cd /workspace/dreamzero && PYTHONPATH=/workspace/dreamzero python test_client_AR.py --port 5000

<!-- source /workspace/cosmos-policy/.venv/bin/activate

source /workspace/.venv/bin/activate -->

uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path /workspace/LIBERO-Cosmos-Policy/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path /workspace/LIBERO-Cosmos-Policy/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path /workspace/LIBERO-Cosmos-Policy/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --task_suite_name libero_10 \
    --local_log_dir experiments/robot/libero/logs/ \
    --randomize_seed False \
    --data_collection False \
    --available_gpus "0,1,2,3,4,5,6,7" \
    --seed 195 \
    --use_variance_scale False \
    --deterministic True \
    --run_id_note chkpt45000--5stepAct--seed195--deterministic \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1

<!-- export HF_ENDPOINT="https://hf-mirror.com"

uvx hf download nvidia/Cosmos-Predict2-2B-Video2World --repo-type model --local-dir /workspace/data1/Cosmos-Predict2-2B-Video2World

hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir /data1/DreamZero-DROID

hf download nvidia/ALOHA-Cosmos-Policy --repo-type dataset --local-dir /data1/lingsheng/ALOHA-Cosmos-Policy

hf download nvidia/Cosmos-Policy-ALOHA-Predict2-2B --repo-type model --local-dir /data1/lingsheng/Cosmos-Policy-ALOHA-Predict2-2B 

hf download Wan-AI/Wan2.1-I2V-14B-480P --repo-type model --local-dir /data1/lingsheng/Wan2.1-I2V-14B-480P

hf download google/umt5-xxl --local-dir /data1/lingsheng/Wan2.1-I2V-14B-480P/umt5-xxl

hf download nvidia/RoboCasa-Cosmos-Policy --repo-type dataset --local-dir /data1/lingsheng/RoboCasa-Cosmos-Policy

hf download nvidia/Cosmos-Policy-RoboCasa-Predict2-2B --repo-type model --local-dir /data1/lingsheng/osmos-Policy-RoboCasa-Predict2-2B 

hf download Qwen/Qwen2-0.5B --local-dir /data1/lingsheng/Qwen2-0.5B
-->

uv sync --extra cu128 --group aloha --python 3.10
# 在 cosmos-policy 的 .venv 中补充 DreamZero 常用依赖（与 cosmos 已有依赖存在版本折中，见上方 .venv 兼容性说明）
uv pip install \
  torch==2.7.0+cu128 \
  torchvision==0.22.0+cu128 \
  triton==3.3.0 \
  -f https://download.pytorch.org/whl/cu128/ \
  torchaudio==2.7.0 \
  pyttsx3==2.90 \
  ray[default]==2.47.1 \
  flask \
  python-socketio>=5.13.0 \
  flask_socketio \
  lmdb \
  meshcat \
  meshcat-shapes \
  rerun-sdk==0.21.0 \
  pygame \
  sshkeyboard \
  msgpack \
  msgpack-numpy \
  pyzmq \
  PyQt6 \
  pin \
  pin-pink \
  timm \
  redis \
  datasets==3.6.0 \
  evdev \
  pybullet \
  gear \
  dm_tree \
  openai \
  tianshou==0.5.1 \
  nvidia-modelopt \
  nvidia-modelopt-core \
  tensorrt \
  openpi-client==0.1.1 \
  huggingface_hub

source /workspace/.venv/bin/activate


  <!-- source /workspace/.venv/bin/activate && uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path /workspace/LIBERO-Cosmos-Policy/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path /workspace/LIBERO-Cosmos-Policy/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path /workspace/LIBERO-Cosmos-Policy/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --task_suite_name libero_10 \
    --local_log_dir experiments/robot/libero/logs/ \
    --randomize_seed False \
    --data_collection False \
    --available_gpus "0,1,2,3,4,5,6,7" \
    --seed 195 \
    --use_variance_scale False \
    --deterministic True \
    --run_id_note chkpt45000--5stepAct--seed195--deterministic \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 -->


python -m cosmos_policy.experiments.robot.aloha.deploy \
  --config cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only \
  --ckpt_path /workspace/data1/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt \
  --config_file cosmos_policy/config/config.py \
  --use_third_person_image True \
  --use_wrist_image True \
  --num_wrist_images 2 \
  --use_proprio True \
  --normalize_proprio True \
  --unnormalize_actions True \
  --dataset_stats_path /workspace/data1/Cosmos-Policy-ALOHA-Predict2-2B/aloha_dataset_statistics.json \
  --t5_text_embeddings_path /workspace/data1/Cosmos-Policy-ALOHA-Predict2-2B/aloha_t5_embeddings.pkl \
  --trained_with_image_aug True \
  --chunk_size 50 \
  --num_open_loop_steps 50 \
  --ar_future_prediction False \
  --ar_value_prediction False \
  --use_jpeg_compression False \
  --flip_images False \
  --num_denoising_steps_action 10 \
  --num_denoising_steps_future_state 1 \
  --num_denoising_steps_value 1 \
  --deterministic True \
  --seed 195


  uv run --extra cu128 --group robocasa --python 3.10 \
 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --num_wrist_images 1 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 \
    --num_open_loop_steps 16 \
    --task_name TurnOffMicrowave \
    --num_trials_per_task 50 \
    --run_id_note chkpt45000--5stepAct--seed195--deterministic \
    --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
    --seed 195 \
    --randomize_seed False \
    --deterministic True \
    --use_variance_scale False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --data_collection False

export WAN21_LOCAL_DIR=/workspace/data1/Wan2.1-I2V-14B-480P





docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/root/.cache \
  -v /home/lingsheng/chennuo/cosmos-policy:/workspace \
  -v /home/lingsheng/chennuo/dreamzero:/workspace/dreamzero \
  -v /data1/lingsheng:/workspace/data1 \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace \
  --entrypoint bash \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  --name cn \
  openpi:installed

    --memory 200g \
  --memory-swap 220g \


apt-get update

apt-get install -y build-essential cmake ninja-build pkg-config \
                   libegl1-mesa-dev libgl1-mesa-dev libx11-dev

bash /workspace/bin/setup_robocasa_env.sh

export WAN21_LOCAL_DIR=/workspace/data1/Wan2.1-I2V-14B-480P

source /workspace/.venv/bin/activate

CUDA_VISIBLE_DEVICES=4 python train_vla_distill.py \
  --student_ckpt_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt \
  --student_config cosmos_policy/config/config.py \
  --student_experiment cosmos_predict2_2b_480p_robocasa_50_demos_per_task \
  --teacher_path /workspace/data1/DreamZero-DROID \
  --output_dir /workspace/outputs/vla_distill_robocasa_retry \
  --robocasa_data_dir /workspace/data1/RoboCasa-Cosmos-Policy/success_only \
  --robocasa_t5_embeddings_path /workspace/data1/RoboCasa-Cosmos-Policy/success_only/t5_embeddings.pkl \
  --max_iterations 45000 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --teacher_tokenizer_path /workspace/data1/umt5-xxl \
  --teacher_image_text_only \
  --train_part adapter \
  --teacher_layer_index 14 \
  --resume_from /workspace/outputs/vla_distill_robocasa_retry/checkpoint_00005500.pt


CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nproc_per_node=4 train_vla_distill.py \
  --student_ckpt_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt \
  --student_config cosmos_policy/config/config.py \
  --student_experiment cosmos_predict2_2b_480p_robocasa_50_demos_per_task \
  --teacher_path /workspace/data1/DreamZero-DROID \
  --output_dir /workspace/outputs/vla_distill_robocasa_retry \
  --robocasa_data_dir /workspace/data1/RoboCasa-Cosmos-Policy/success_only \
  --robocasa_t5_embeddings_path /workspace/data1/RoboCasa-Cosmos-Policy/success_only/t5_embeddings.pkl \
  --max_iterations 45000 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --teacher_tokenizer_path /workspace/data1/umt5-xxl \
  --teacher_image_text_only \
  --train_part adapter \
  --teacher_layer_index 14 \
  --resume_from /workspace/outputs/vla_distill_robocasa_retry/checkpoint_00005500.pt

source /home/lingsheng/jiangyuhua/miniconda/etc/profile.d/conda.sh

pip install -U dm-tree


cd /home/lingsheng/chennuo/cosmos-policy

python3 scripts/plot_loss_curve.py \
  --csv outputs/vla_distill_robocasa/loss_curve.csv \
  --out outputs/vla_distill_robocasa/loss_curve.png


<!-- uv sync --extra cu128 --group robocasa  --python 3.10

source /workspace/.venv/bin/activate

uv pip install -e robocasa-cosmos-policy

uv pip install \
  torch==2.7.0+cu128 \
  torchvision==0.22.0+cu128 \
  triton==3.3.0 \
  -f https://download.pytorch.org/whl/cu128/ \
  torchaudio==2.7.0 \
  pyttsx3==2.90 \
  ray[default]==2.47.1 \
  flask \
  python-socketio>=5.13.0 \
  flask_socketio \
  lmdb \
  meshcat \
  meshcat-shapes \
  rerun-sdk==0.21.0 \
  pygame \
  sshkeyboard \
  msgpack \
  msgpack-numpy \
  pyzmq \
  PyQt6 \
  pin \
  pin-pink \
  timm \
  redis \
  datasets==3.6.0 \
  evdev \
  pybullet \
  gear \
  dm_tree \
  openai \
  tianshou==0.5.1 \
  nvidia-modelopt \
  nvidia-modelopt-core \
  tensorrt \
  openpi-client==0.1.1 \
  huggingface_hub

uv pip install "numpy>=1.26.0,<2" --python /workspace/.venv/bin/python

uv pip install --reinstall numba --python /workspace/.venv/bin/python -->



# 若在 scaler.step(optimizer) 时 OOM，可先设：
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 训练脚本已在 step 前调用 torch.cuda.empty_cache() 缓解碎片；仍 OOM 可加大 grad_accumulation_steps、减小 batch 或启用 --teacher_image_text_only 省教师显存。

uv run --extra cu128 --group robocasa --python 3.10 


export COSMOS_ADAPTER_CONTEXT_SCALE=1

python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval_guided \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt \
    --config_file cosmos_policy/config/config.py \
    --teacher_path /workspace/data1/DreamZero-DROID \
    --distill_ckpt_path /workspace/outputs/vla_distill_robocasa_retry/checkpoint_00005000.pt \
    --adapter_only_from_distill true \
    --student_experiment cosmos_predict2_2b_480p_robocasa_50_demos_per_task \
    --teacher_tokenizer_path /workspace/data1/umt5-xxl \
    --dataset_stats_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --use_wrist_image True \
    --num_wrist_images 1 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --chunk_size 32 \
    --num_open_loop_steps 16 \
    --task_name TurnOffMicrowave \
    --num_trials_per_task 5 \
    --local_log_dir cosmos_policy/experiments/robot/robocasa/logs_guided/ \
    --seed 195 \
    --deterministic True \
    --num_denoising_steps_action 5 \
    --mujoco_gl egl \
    --data_collection False \
    --teacher_block_profile true


python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --num_wrist_images 1 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 \
    --num_open_loop_steps 16 \
    --task_name TurnOffMicrowave \
    --num_trials_per_task 50 \
    --run_id_note chkpt45000--5stepAct--seed195--deterministic \
    --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
    --seed 195 \
    --randomize_seed False \
    --deterministic True \
    --use_variance_scale False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --data_collection False

python scripts/run_robocasa_full_benchmark.py \
  --suite cosmos24 \
  --num-trials-per-task 50 \
  --seeds 195,196,197 \
  --output-json robocasa_benchmark_summary.json \
  -- \
  -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
  --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
  --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
  --config_file cosmos_policy/config/config.py \
  --use_wrist_image True \
  --num_wrist_images 1 \
  --use_proprio True \
  --normalize_proprio True \
  --unnormalize_actions True \
  --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
  --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
  --trained_with_image_aug True \
  --chunk_size 32 \
  --num_open_loop_steps 16 \
  --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
  --deterministic True \
  --randomize_seed False \
  --use_variance_scale False \
  --use_jpeg_compression True \
  --flip_images True \
  --num_denoising_steps_action 5 \
  --num_denoising_steps_future_state 1 \
  --num_denoising_steps_value 1 \
  --data_collection False