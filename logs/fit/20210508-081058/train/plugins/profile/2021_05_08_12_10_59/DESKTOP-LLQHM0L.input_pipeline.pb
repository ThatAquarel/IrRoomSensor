	?6?[`@?6?[`@!?6?[`@	5jx?{??5jx?{??!5jx?{??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?6?[`@??k	????A??s?{@Y?1w-!??*	?????I@2F
Iterator::Model???Mb??!?+???sG@);?O??n??1???i??A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF??_??!yDR???7@)?j+??݃?1i???|\3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg??j+???!y?oqZ7@)?J?4??1?]?/7?0@:Preprocessing2U
Iterator::Model::ParallelMapV2?I+?v?!O??+??%@)?I+?v?1O??+??%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceF%u?k?!`n??X@)F%u?k?1`n??X@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?]K?=??!?=?J@)?~j?t?h?1(????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!@?ZV??@)/n??b?1@?ZV??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no95jx?{??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??k	??????k	????!??k	????      ??!       "      ??!       *      ??!       2	??s?{@??s?{@!??s?{@:      ??!       B      ??!       J	?1w-!???1w-!??!?1w-!??R      ??!       Z	?1w-!???1w-!??!?1w-!??JCPU_ONLYY5jx?{??b 