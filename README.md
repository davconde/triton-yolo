# triton-yolo
Configuration and tools for deploying YOLO on Triton Inference Server.

## Prepare the Triton Inference Server container
Pull an image from the NVIDIA NGC Catalog:

```bash
TRITON_VER=25.03

docker pull nvcr.io/nvidia/tritonserver:${TRITON_VER}-py3
```

Clone this repository and access it:

```bash
git clone https://github.com/davconde/triton-yolo.git
cd triton-yolo
```

Launch the container with the following command:

```bash
TRITON_VER=25.03

export DISPLAY=:0
xhost +
docker run -it --rm --shm-size=256m --runtime=nvidia --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $(pwd):/triton-yolo -w /triton-yolo nvcr.io/nvidia/tritonserver:${TRITON_VER}-py3
```

## Prepare the model repository
Move the ONNX file of the model to `model_repository/yolo/1/model.onnx`. If required, this ONNX file can be generated from the PyTorch model on the same machine/environment where it was trained by using [utils/export_yolo11.py](utils/export_yolo11.py) as follows:

```bash
RESOLUTION=640

python3 utils/export_yoloV11.py -w model.pt --dynamic --simplify -s ${RESOLUTION}
```

Build the TensorRT model from the ONNX file (only on NVIDIA hardware):

```bash
ONNX_PATH=./model_repository/yolo/1/model.onnx
TRT_PATH=./model_repository/yolo/2/model.plan
RESOLUTION=640
CHANNELS=3
INPUT_LAYER=input
MIN_BATCH=1
OPT_BATCH=4
MAX_BATCH=32

/usr/src/tensorrt/bin/trtexec --onnx=${ONNX_PATH} --saveEngine=${TRT_PATH} --minShapes=${INPUT_LAYER}:${MIN_BATCH}x${CHANNELS}x${RESOLUTION}x${RESOLUTION} --optShapes=${INPUT_LAYER}:${OPT_BATCH}x${CHANNELS}x${RESOLUTION}x${RESOLUTION} --maxShapes=${INPUT_LAYER}:${MAX_BATCH}x${CHANNELS}x${RESOLUTION}x${RESOLUTION} --shapes=${INPUT_LAYER}:5x${CHANNELS}x${RESOLUTION}x${RESOLUTION}
```

## Launch the server

Run the following command inside the container:

```bash
# Use the TensorRT backend (only available on NVIDIA GPUs)
tritonserver --model-repository=model_repository --model-config-name=tensorrt

# Use the ONNX runtime (it will take time to warmup after first client connection)
tritonserver --model-repository=model_repository --model-config-name=onnx
```
