# TIAMAT-ROS2-Controller

# Prerequisites

This repo depends on the AWS-ECR setup mentioned in the DARPA APSU challenge documentation. 
Setup AWS account following instructions mentioned in the AWS account setup section in the documentation. 


## Pulling the `tiamat` simulator container

Configure the AWS CLI 

```bash
aws configure

AWS Access Key ID [None]: <Your Access Key>
AWS Secret Access Key [None]: <Your Secret Access Key>
Default region name [None]: us-gov-west-1
Default output format [None]:
```

Pulling and tagging tiamat-baseline image
```bash
docker pull 276226212704.dkr.ecr.us-gov-west-1.amazonaws.com/tiamat/baseline:latest
docker tag 276226212704.dkr.ecr.us-gov-west-1.amazonaws.com/tiamat/baseline tiamat
```

## Debugging: User does not belong to the docker group 

```bash
groups
```

Add user to the docker group.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Debugging: Multiple users using the same host machine 

Multiple users running on the same host machine may interfere with ROS2 messages. Assign an arbitary number as the ROS_DOMAIN_ID 
in both simulator and agent environment.

```bash
export ROS_DOMAIN_ID=<SOME_RANDOM_NUMBER>
```

# AWS Docker Container

## Download datasets

Download [habitat-dataset.zip](https://drive.google.com/file/d/1ry0TAV5yb5Kc6E0iQGMoCgAHpZMo0aHh/view?usp=sharing) from Google Drive and unzip in your host machine. 

Alternatively if you have setup habitat-sim in your host machine, run 

```bash
python -m habitat_sim.utils.datasets_download --replace --uids \
habitat_test_scenes habitat_example_objects hab3_bench_assets hab_spot_arm ycb \
--data-path /root/habitat-lab/data
```

## Run TIAMAT-AWS docker container 

Do not manually setup `DISPLAY` variable as your `DISPLAY` may not be `:0` given your setup. 

```bash
xhost +local:docker

export DISPLAY=$(echo $DISPLAY)

docker run -it --rm --name tiamat-sim --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e PRIVACY_CONSENT=Y \
    -e DISPLAY=$DISPLAY \
    -e ACCEPT_EULA=Y \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data \
    -v ~/docker/isaac-sim/documents:/root/Documents \
    --entrypoint /bin/bash \
    tiamat
```

## Add datasets to the container and build simlinks

```bash
docker cp <your-dataset-dir> tiamat-sim:/root/habitat-lab/
docker exec tiamat-sim bash -lc '
ln -sf /root/habitat-lab/data/hab3_bench_assets /root/habitat_environment/sim/data/ &&
ln -sf /root/habitat-lab/data/objects/ycb /root/habitat_environment/sim/data/objects/ &&
ln -sf /root/habitat-lab/data/robots/hab_spot_arm /root/habitat_environment/sim/data/robots/ &&
ln -sf /root/habitat-lab/data/scene_datasets /root/habitat_environment/sim/data/
'
```

### Running tiamat simulator container (Within the interactive shell)

```bash
conda activate tiamat-habitat
python habitat_environment/sim/tiamat_runner.py --smoke --headless
```

# Agent Docker Container (UF)

## Cloning this repository

Clone this repository to your `<repository_location>`

## Building the dev image
```bash
nohup docker build -t ros2-humble-conda-cuda12.1 . > out.txt 2>&1
```

## Run agent Docker container

```bash
xhost +local:docker

export DISPLAY=$(echo $DISPLAY)

docker run -it --rm \
    --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e ROS_STATIC_PEERS=tiamat \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    -v <repository_location>:/workspace \
    ros2-humble-conda-cuda12.1
```

## Source conda
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate ros2
cd /workspace
python controller.py
```

### Debugging

```bash
docker exec -it <container_name_or_id> /bin/bash
ros2 topic list
```

## RTAB Map Launch

In a new ROS2 sourced shell run the following. 

```bash
cd /workspace/launch
ros2 launch rtabmap_launch.py
```

## Nav2 Launch

In a new ROS2 sourced shell run the following. 

```bash
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=true \
    params_file:=/workspace/configs/nav2_params.yaml
```

<!-- ros2  run teleop_twist_keyboard teleop_twist_keyboard cmd_vel:/spot/cmd_vel -->
