# !/bin/bash
echo "Installing in $(which python)"

pushd code/torchvision
conda install -y -c conda-forge libjpeg-turbo
conda install -y pillow=10.3.0
python setup.py install
popd
echo "Finished!"