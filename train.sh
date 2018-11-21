if [ $ARCHITECTURE == "mobilenet_1.0_${IMAGE_SIZE}" ]
then
echo "running mobilenet_1.0_${IMAGE_SIZE}"
python -m scripts.retrain \
	--bottleneck_dir=tf_files/bottlenecks \
	--how_many_training_steps=4000 \
	--model_dir=tf_files/models/"${ARCHITECTURE}"_"${LEARNING_RATE}" \
	--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}"_"${LEARNING_RATE}" \
	--output_graph=tf_files/retrained_graph_"${ARCHITECTURE}"_"${LEARNING_RATE}".pb \
	--output_labels=tf_files/retrained_labels_"${ARCHITECTURE}"_"${LEARNING_RATE}".txt \
	--architecture="${ARCHITECTURE}" \
	--image_dir=tf_files/structure_photos/train \
	--learning_rate=${LEARNING_RATE}
fi

if [ $ARCHITECTURE == "inception_v3" ]
then
echo "running inception_v3"
python -m scripts.retrain \
	--bottleneck_dir=tf_files/bottlenecks \
	--how_many_training_steps=4000 \
	--model_dir=tf_files/models/"${ARCHITECTURE}"_"${LEARNING_RATE}" \
	--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}"_"${LEARNING_RATE}" \
	--output_graph=tf_files/retrained_graph_"${ARCHITECTURE}"_"${LEARNING_RATE}".pb \
	--output_labels=tf_files/retrained_labels_"${ARCHITECTURE}"_"${LEARNING_RATE}".txt \
	--architecture="${ARCHITECTURE}" \
	--image_dir=tf_files/structure_photos/train \
	--learning_rate=${LEARNING_RATE} \
	--input_layer=Mul \
	--input_width=299 \
	--input_height=299
fi
