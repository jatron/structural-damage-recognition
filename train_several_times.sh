export IMAGE_SIZE=224
export ARCHITECTURE="mobilenet_1.0_${IMAGE_SIZE}"

export LEARNING_RATE=0.01

./train.sh

export LEARNING_RATE=0.001

./train.sh

export LEARNING_RATE=0.0001

./train.sh
