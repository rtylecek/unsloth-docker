# Clone and build
git clone --depth=1 https://github.com/facebookresearch/xformers --recursive
cd xformers
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py install
cd ..
rm -rf xformers
