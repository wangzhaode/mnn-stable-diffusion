# Stable Diffusion MNN

## Usage

### 1. Compile MNN library
```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir build
cmake -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ..
make -j8
cp libMNN.so express/libMNN_Express.so tools/cv/libMNNOpenCV.so /path/to/stable-diffusion-mnn/libs
```

### 2. Download Models
Download models from `github release` to `/path/to/stable-diffusion-mnn/resource`
```bash
cd resource
wget https://github.com/wangzhaode/stable-diffusion-mnn/releases/download/v0.1/text_encoder.mnn
wget https://github.com/wangzhaode/stable-diffusion-mnn/releases/download/v0.1/vae_decoder.mnn
wget https://github.com/wangzhaode/stable-diffusion-mnn/releases/download/v0.1/unet.mnn
```

### 2. Build and Run
```bash
mkdir build
cd build
cmake ..
make -j4
./main "飞流直下三千尺，疑是银河落九天，唐诗，水墨，国画。" demo.jpg
[##################################################]  [100%] [iter time: 411.441000 ms]
SUCCESS! write to demo.jpg
```
![demo.jpg](https://github.com/wangzhaode/stable-diffusion-mnn/blob/main/resource/demo.jpg)

# Ref
https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1
