	???Q?@???Q?@!???Q?@	??R?y????R?y??!??R?y??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???Q?@v??????Ad]?F?@Y?(??0??*	33333?F@2F
Iterator::Model䃞ͪϕ?!??3??3G@)??ܵ?|??1Qm4?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?HP???!ΰ?w[?:@)??ׁsF??1??????5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?(??0??!w!?v!?J@)?I+?v?1?hL??'@:Preprocessing2U
Iterator::Model::ParallelMapV2??_?Lu?!????o?&@)??_?Lu?1????o?&@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaplxz?,C|?!f?f?.@);?O??nr?1??????#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicea2U0*?c?!??????@)a2U0*?c?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!?|`?'@)HP?s?b?1?|`?'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??R?y??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v??????v??????!v??????      ??!       "      ??!       *      ??!       2	d]?F?@d]?F?@!d]?F?@:      ??!       B      ??!       J	?(??0???(??0??!?(??0??R      ??!       Z	?(??0???(??0??!?(??0??JCPU_ONLYY??R?y??b 