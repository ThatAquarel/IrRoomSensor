	??<,?Z@??<,?Z@!??<,?Z@	??Љ?\????Љ?\??!??Љ?\??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??<,?Z@??%䃞??A??#???@Y??e?c]??*	33333sB@2F
Iterator::Model	?^)ː?!??8??8F@)?g??s???1?I???<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?+e?X??!?On??>@)U???N@??1?6A?hy9@:Preprocessing2U
Iterator::Model::ParallelMapV2?????w?!zPL?o/@)?????w?1zPL?o/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v??z?!H-/|?1@)HP?s?r?1??????(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipˡE?????!r?q?K@)??_?Le?1?&H-/@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!Y?7?"?@)????Mb`?1Y?7?"?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!_?!??4???@)ŏ1w-!_?1??4???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??Љ?\??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??%䃞????%䃞??!??%䃞??      ??!       "      ??!       *      ??!       2	??#???@??#???@!??#???@:      ??!       B      ??!       J	??e?c]????e?c]??!??e?c]??R      ??!       Z	??e?c]????e?c]??!??e?c]??JCPU_ONLYY??Љ?\??b 