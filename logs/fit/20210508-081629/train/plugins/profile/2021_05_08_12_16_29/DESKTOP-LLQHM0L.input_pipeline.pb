	H?z?G@H?z?G@!H?z?G@	 *?3?? *?3??! *?3??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$H?z?G@s??A??A??ʡ?@Y7?[ A??*	     ?T@2F
Iterator::Model"??u????!???k?D@)V-???1D.+JxA@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?HP???!N?_{?e=@)??A?f??1i???C.9@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??+e???!\V??FM@)?HP???1N?_{?e-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?~j?t???!???h?,@)n????11P?M?'@:Preprocessing2U
Iterator::Model::ParallelMapV2??_vOv?!??C.+@)??_vOv?1??C.+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey?&1?l?!?ˊ??@)y?&1?l?1?ˊ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!f?@	o4@)/n??b?1f?@	o4@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9 *?3??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s??A??s??A??!s??A??      ??!       "      ??!       *      ??!       2	??ʡ?@??ʡ?@!??ʡ?@:      ??!       B      ??!       J	7?[ A??7?[ A??!7?[ A??R      ??!       Z	7?[ A??7?[ A??!7?[ A??JCPU_ONLYY *?3??b 