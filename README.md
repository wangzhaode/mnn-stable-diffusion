# Stable Diffusion MNN

## Usage

### 1. Compile MNN library
```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir build
cmake -DMNN_BUILD_OPENCV=ON MNN_IMGCODECS=ON ..
make -j8
cp libMNN.so express/libMNN_Express.so tools/cv/libMNNOpenCV.so /path/to/stable-diffusion-mnn/libs
```

### 2. Build and Run
```bash
mkdir build
cd build
cmake ..
make -j4
./main "飞流直下三千尺，油画" f.jpg
```

# Ref
https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1
