?	??\@??\@!??\@	????????????????!????????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??\@???9#J??AX?5?;?@Y???{????*	?????H@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat+??????!Yl:Yl:D@)L7?A`???1$?$?A@:Preprocessing2F
Iterator::ModelU???N@??!??????C@)S?!?uq??10?/?;@:Preprocessing2U
Iterator::Model::ParallelMapV2??_vOv?!h~h&@)??_vOv?1h~h&@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??H?}}?!?!??!?-@)n??t?1??T??T$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!?q??q?@)?~j?t?h?1?q??q?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u???!	xxN@)??_vOf?1h~h@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceHP?s?b?!?N?N@)HP?s?b?1?N?N@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???9#J?????9#J??!???9#J??      ??!       "      ??!       *      ??!       2	X?5?;?@X?5?;?@!X?5?;?@:      ??!       B      ??!       J	???{???????{????!???{????R      ??!       Z	???{???????{????!???{????JCPU_ONLYY????????b Y      Y@q?O?B!?!@"?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 