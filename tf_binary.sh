unamestr=`uname`
if [[ "$unamestr" == "Darwin" ]]; then
	echo "MAC CPU"
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
else
	echo "Linux GPU"
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl
fi
