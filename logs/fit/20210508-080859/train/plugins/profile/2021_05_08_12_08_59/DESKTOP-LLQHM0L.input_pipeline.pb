	W[????@W[????@!W[????@	S5?3{???S5?3{???!S5?3{???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$W[????@X9??v??A?J??@Y?@??ǘ??*	?????T@2F
Iterator::Model o?ŏ??! ?[pPbE@)??H?}??1??<??A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??A?f??!C?ZR:@)U???N@??1]??\?p7@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9??m4???!?????L@)X?5?;N??1?p߂?5@:Preprocessing2U
Iterator::Model::ParallelMapV2?I+?v?!2ˢqn@)?I+?v?12ˢqn@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ǘ????!q߂?3$@);?O??nr?1(1ˢq@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??H?}m?!??<??@)??H?}m?1??<??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!Y?????@)?J?4a?1Y?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9T5?3{???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	X9??v??X9??v??!X9??v??      ??!       "      ??!       *      ??!       2	?J??@?J??@!?J??@:      ??!       B      ??!       J	?@??ǘ???@??ǘ??!?@??ǘ??R      ??!       Z	?@??ǘ???@??ǘ??!?@??ǘ??JCPU_ONLYYT5?3{???b 