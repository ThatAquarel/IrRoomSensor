	?-???@?-???@!?-???@	?M?3$g???M?3$g??!?M?3$g??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?-???@?+e?X??A?HP?@Y????Mb??*	?????YO@2F
Iterator::Model?Zd;??!m<??yRH@)?
F%u??1??=?SAD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat
ףp=
??!???*M?A@)????<,??1???ak?@:Preprocessing2U
Iterator::Model::ParallelMapV2??ZӼ?t?!?U???D @)??ZӼ?t?1?U???D @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??H?}}?!H??m?&@)??ZӼ?t?1?U???D @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!q????@)Ǻ???f?1q????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ܵ?|??!??T??I@){?G?zd?1????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?J?4a?!*~I?T?
@)?J?4a?1*~I?T?
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?M?3$g??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?+e?X???+e?X??!?+e?X??      ??!       "      ??!       *      ??!       2	?HP?@?HP?@!?HP?@:      ??!       B      ??!       J	????Mb??????Mb??!????Mb??R      ??!       Z	????Mb??????Mb??!????Mb??JCPU_ONLYY?M?3$g??b 