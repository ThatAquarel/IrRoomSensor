	???镲@???镲@!???镲@	??߹??????߹????!??߹????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???镲@???N@??Aj?t?@Y?l??????*	?????LD@2F
Iterator::ModelDio??ɔ?!      I@)???Q???1???dyB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??_?L??!????9@)?St$????1?Δ^q4@:Preprocessing2U
Iterator::Model::ParallelMapV2?g??s?u?!tl?*@)?g??s?u?1tl?*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS?!?uq{?!?.1k??0@)/n??r?1????Ĭ%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDio??ɔ?!      I@)-C??6j?1C???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceHP?s?b?!??|7??@)HP?s?b?1??|7??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!??=??@)?J?4a?1??=??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??߹????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???N@?????N@??!???N@??      ??!       "      ??!       *      ??!       2	j?t?@j?t?@!j?t?@:      ??!       B      ??!       J	?l???????l??????!?l??????R      ??!       Z	?l???????l??????!?l??????JCPU_ONLYY??߹????b 