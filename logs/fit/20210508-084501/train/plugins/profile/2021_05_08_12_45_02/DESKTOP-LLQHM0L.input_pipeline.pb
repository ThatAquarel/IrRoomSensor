	P??nC@P??nC@!P??nC@	??G#f?????G#f???!??G#f???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$P??nC@?????K??A?8??m?@Y_)?Ǻ??*	gfffffH@2F
Iterator::Model?(??0??!).?u4I@)??d?`T??1????WB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS?!?uq??!?).?u;@)A??ǘ???12?h??6@:Preprocessing2U
Iterator::Model::ParallelMapV2S?!?uq{?!?).?u+@)S?!?uq{?1?).?u+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? ?	??!??%C??/@)??ZӼ?t?1?d???$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??@??ǘ?!???}??H@)a??+ei?1T\2?h@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??_?Le?!?:ڼO@)??_?Le?1?:ڼO@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!??:?@)HP?s?b?1??:?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??G#f???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????K???????K??!?????K??      ??!       "      ??!       *      ??!       2	?8??m?@?8??m?@!?8??m?@:      ??!       B      ??!       J	_)?Ǻ??_)?Ǻ??!_)?Ǻ??R      ??!       Z	_)?Ǻ??_)?Ǻ??!_)?Ǻ??JCPU_ONLYY??G#f???b 