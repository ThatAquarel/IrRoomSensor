	?:pΈ?@?:pΈ?@!?:pΈ?@	?m??59???m??59??!?m??59??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?:pΈ?@u?V??A??o_?@Yz6?>W[??*	fffff?F@2F
Iterator::Model????<,??!??mF??E@)%u???1ϻxL@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!??e??S9@)a2U0*???1(?nY??4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??0?*??!????9@)Έ?????1?6??@N4@:Preprocessing2U
Iterator::Model::ParallelMapV2{?G?zt?!?H???%@){?G?zt?1?H???%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}гY????!r??R~L@)??_vOf?1???d??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice{?G?zd?!?H???@){?G?zd?1?H???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!!:ܟ?w@)????Mb`?1!:ܟ?w@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?m??59??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	u?V??u?V??!u?V??      ??!       "      ??!       *      ??!       2	??o_?@??o_?@!??o_?@:      ??!       B      ??!       J	z6?>W[??z6?>W[??!z6?>W[??R      ??!       Z	z6?>W[??z6?>W[??!z6?>W[??JCPU_ONLYY?m??59??b 