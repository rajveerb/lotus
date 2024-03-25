| Transformation         | Function                           | Library       |
|------------------------|------------------------------------|---------------|
| ToTensor                | futex_wake                          | libgomp.so.1  |
|                         | gomp_simple_barrier_wait            | libgomp.so.1  |
|                         | func@0xce88e0                       | libtorch_cpu.so |
|                         | gomp_team_barrier_wait_end          | libgomp.so.1  |
|                         | _Py_NewReference                    | python3.10    |
|                         | ImagingRawEncode                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _PyObject_GetMethod                 | python3.10    |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                         | tupledealloc                        | python3.10    |
|                         | lookdict_unicode_nodummy            | python3.10    |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::TensorIteratorBase::loop_2d_from_1d<at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1}>(, signed char, at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1} const&)::{lambda(char**long const*, long, long)#1}> | libtorch_cpu.so |
|                         | __GI___libc_malloc                  | libc.so.6     |
|                         | _PyBytes_Resize                     | python3.10    |
|                         | gomp_team_end                       | libgomp.so.1  |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#2}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#2}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | PyType_HasFeature                   | python3.10    |
|                         | arena_map_get                       | python3.10    |
|                         | do_mkvalue                          | python3.10    |
|                         | munmap                              | libc.so.6     |
|                         | PyBuffer_Release                    | python3.10    |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(unsigned char)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<unsigned char>)#2}>> | libtorch_cpu.so |
|                         | _Py_INCREF                          | python3.10    |
|                         | ImagingPackRGB                      | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __GI_                               | libc.so.6     |
|                        |                                    |               |
| RandomErasing           | __tls_get_addr                      | ld-linux-x86-64.so.2 |
|                         | gomp_simple_barrier_wait            | libgomp.so.1  |
|                         | gomp_team_barrier_wait_end          | libgomp.so.1  |
|                         | _int_free                           | libc.so.6     |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(float)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | gomp_team_end                       | libgomp.so.1  |
|                         | torch::autograd::THPVariable_item   | libtorch_python.so |
|                        |                                    |               |
| RandomAutoContrast      | im_point_3x8_3x8                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingGetHistogram                 | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __libc_calloc                       | libc.so.6     |
|                         | _int_free                           | libc.so.6     |
|                        |                                    |               |
| PILToTensor             | PyBuffer_Release                    | python3.10    |
|                         | object_dealloc                      | python3.10    |
|                         | PyType_HasFeature                   | python3.10    |
|                         | __GI___libc_malloc                  | libc.so.6     |
|                         | _PyEval_EvalFrameDefault            | python3.10    |
|                         | stringlib_bytes_join                | python3.10    |
|                         | tuple_alloc                         | python3.10    |
|                         | _int_free                           | libc.so.6     |
|                         | _PyObject_GC_TRACK                  | python3.10    |
|                         | _PyObject_GetMethod                 | python3.10    |
|                         | ImagingPackRGB                      | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingRawEncode                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __GI_                               | libc.so.6     |
|                         | _Py_DECREF                          | python3.10    |
|                         | PyObject_IsTrue                     | python3.10    |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                         | do_mkvalue                          | python3.10    |
|                         | munmap                              | libc.so.6     |
|                        |                                    |               |
| RandomPerspective       | perspective_transform               | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | bilinear_filter32RGB                | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _int_free                           | libc.so.6     |
|                         | ImagingFill                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingGenericTransform             | _imaging.cpython-310-x86_64-linux-gnu.so |
|                        |                                    |               |
| ToPILImage              | FLOAT_multiply_FMA3__AVX2           | _multiarray_umath.cpython-310-x86_64-linux-gnu.so |
|                         | PyArray_ToString                    | _multiarray_umath.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingUnpackRGB                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingFill                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _aligned_contig_cast_float_to_ubyte | _multiarray_umath.cpython-310-x86_64-linux-gnu.so |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                         | __memset_avx2_erms                  | libc.so.6     |
|                         | munmap                              | libc.so.6     |
|                        |                                    |               |
| ConvertImageDType       | _PyType_Lookup                      | python3.10    |
|                        |                                    |               |
| RandomCrop              | _int_free                           | libc.so.6     |
|                        |                                    |               |
| Grayscale               | rgb2l                               | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _int_free                           | libc.so.6     |
|                         | convert                             | _imaging.cpython-310-x86_64-linux-gnu.so |
|                        |                                    |               |
| RandomHorizontalFlip    | _int_free                           | libc.so.6     |
|                         | ImagingFlipLeftRight                | _imaging.cpython-310-x86_64-linux-gnu.so |
|                        |                                    |               |
| ColorJitter             | __round                             | libm.so.6     |
|                         | hsv2rgb                             | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingMerge                        | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _int_free                           | libc.so.6     |
|                         | ImagingGetHistogram                 | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __libc_calloc                       | libc.so.6     |
|                         | rgb2hsv                             | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                         | __memset_avx2_erms                  | libc.so.6     |
|                         | ImagingBlend                        | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __GI___libc_malloc                  | libc.so.6     |
|                         | convert                             | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | stringlib_bytes_join                | python3.10    |
|                         | l2rgb                               | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _PyBytes_Resize                     | python3.10    |
|                         | UBYTE_add_AVX2                      | _multiarray_umath.cpython-310-x86_64-linux-gnu.so |
|                         | rgb2l                               | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingSplit                        | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | object_dealloc                      | python3.10    |
|                         | pymalloc_free                       | python3.10    |
|                         | _PyFunction_Vectorcall              | python3.10    |
|                         | lookdict_unicode                    | python3.10    |
|                         | _PyDict_LoadGlobal                  | python3.10    |
|                         | __ieee754_fmod                      | libm.so.6     |
|                         | _Py_INCREF                          | python3.10    |
|                         | ImagingFill                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | rgb2hsv_row                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | arena_map_is_used                   | python3.10    |
|                         | _Py_SET_REFCNT                      | python3.10    |
|                         | memcpy                              | python3.10    |
|                         | __fmod                              | libm.so.6     |
|                        |                                    |               |
| TenCrop                 | _PyObject_GC_UNTRACK                | python3.10    |
|                         | _int_free                           | libc.so.6     |
|                         | _PyEval_EvalFrameDefault            | python3.10    |
|                         | ImagingFlipLeftRight                | _imaging.cpython-310-x86_64-linux-gnu.so |
|                        |                                    |               |
| RandomEqualize          | im_point_3x8_3x8                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingGetHistogram                 | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __libc_calloc                       | libc.so.6     |
|                         | _int_free                           | libc.so.6     |
|                        |                                    |               |
| RandomRotation          | _PyEval_EvalFrameDefault            | python3.10    |
|                         | _int_free                           | libc.so.6     |
|                         | ImagingFill                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _Py_INCREF                          | python3.10    |
|                         | ImagingTransformAffine              | _imaging.cpython-310-x86_64-linux-gnu.so |
|                        |                                    |               |
| Normalize               | futex_wake                          | libgomp.so.1  |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#2}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#2}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | gomp_simple_barrier_wait            | libgomp.so.1  |
|                         | gomp_team_barrier_wait_end          | libgomp.so.1  |
|                         | gomp_team_end                       | libgomp.so.1  |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(float)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | munmap                              | libc.so.6     |
|                        |                                    |               |
| Pad                     | ImagingFill                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                        |                                    |               |
| RandomSolarize          | im_point_3x8_3x8                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __libc_calloc                       | libc.so.6     |
|                         | _int_free                           | libc.so.6     |
|                        |                                    |               |
| ElasticTransform        | c10::TensorImpl::requires_grad      | libc10.so     |
|                         | futex_wake                          | libgomp.so.1  |
|                         | at::TensorIteratorBase::serial_for_each | libtorch_cpu.so |
|                         | gomp_simple_barrier_wait            | libgomp.so.1  |
|                         | at::TensorIteratorBase::compute_strides | libtorch_cpu.so |
|                         | gomp_team_barrier_wait_end          | libgomp.so.1  |
|                         | _int_free                           | libc.so.6     |
|                         | std::_Function_handler<void (long, long, long), dnnl::impl::cpu::x64::jit_uni_reorder_t::omp_driver_3d(int, int, int, char const*, char*, float const*, float const*, int, int, int*) const::{lambda(longlong, long)#1}>::_M_invoke | libtorch_cpu.so |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::mul_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::mul_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | at::native::(anonymous namespace)::grid_sample_2d_grid_slice_iterator<float, at::native::(anonymous namespace)::grid_sampler_2d_cpu_kernel_impl(at::TensorBase const&, at::TensorBase const&, at::TensorBase const&, long, long, bool)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(longlong)#10}::operator()(long, long) const::{lambda(at::vec::AVX2::Vectorized<float> const&at::vec::AVX2::Vectorized<float> const&, long, long)#1}> | libtorch_cpu.so |
|                         | jit_avx2_conv_fwd_kernel_f32        | [Dynamic code] |
|                         | dnnl::impl::cpu::x64::jit_uni_reorder_t::omp_driver_3d | libtorch_cpu.so |
|                         | dnnl::impl::cpu::x64::jit_uni_reorder_t::fill_curr_data_chunks | libtorch_cpu.so |
|                         | torch::autograd::isFwGradDefined.part.0 | libtorch_cpu.so |
|                         | __GI___libc_malloc                  | libc.so.6     |
|                         | jit_uni_reorder_kernel_f32          | [Dynamic code] |
|                         | gomp_team_end                       | libgomp.so.1  |
|                         | func@0xcf7de0                       | libtorch_cpu.so |
|                         | __tls_get_addr                      | ld-linux-x86-64.so.2 |
|                         | at::get_thread_num                  | libtorch_cpu.so |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}, at::native::(anonymous namespace)::fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#2}>> | libtorch_cpu.so |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#2}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#2}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | __posix_memalign                    | libc.so.6     |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::TensorIteratorBase::loop_2d_from_1d<at::native::AVX2::cpu_serial_kernel<at::native::templates::cpu::(anonymous namespace)::uniform_kernel<at::CPUGeneratorImpl*>(void, at::TensorIteratorBase&, double, double, at::CPUGeneratorImpl*)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda()#1}&>(void, at::TensorIteratorBase&, at::CPUGeneratorImpl*&&, at::Range const&)::{lambda(char**long const*, long)#1}>(, signed char, at::CPUGeneratorImpl* const&)::{lambda(char**long const*, long, long)#1}> | libtorch_cpu.so |
|                         | torch::PythonArgs::symintlist       | libtorch_python.so |
|                         | c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reset_ | libtorch_cpu.so |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | [Unknown stack frame(s)]            | [Unknown]     |
|                         | at::CPUGeneratorImpl::random        | libtorch_cpu.so |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(float)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | munmap                              | libc.so.6     |
|                         | dnnl::impl::convolution_fwd_pd_t::src_md | libtorch_cpu.so |
|                         | unicode_eq                          | python3.10    |
|                         | _ZN2at8internal15invoke_parallelIZNS_6native12_GLOBAL__N_111cpu_paddingIfNS3_13ReflectionPadEEEvRKNS_6TensorES8_RNS3_13PaddingParamsEEUlllE1_EEvlllRKT_._omp_fn.0 | libtorch_cpu.so |
|                         | at::detail::empty_cpu               | libtorch_cpu.so |
|                         | func@0xcd04a0                       | libtorch_cpu.so |
|                         | dnnl::impl::cpu::x64::jit_avx2_convolution_fwd_t::execute_forward(dnnl::impl::exec_ctx_t const&) const::{lambda(intint)#1}::operator() | libtorch_cpu.so |
|                         | __libc_open64                       | libpthread.so.0 |
|                        |                                    |               |
| RandomInvert            | im_point_3x8_3x8                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingAllocateArray                | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __libc_calloc                       | libc.so.6     |
|                         | _int_free                           | libc.so.6     |
|                        |                                    |               |
| RandomResizedCrop       | ImagingResampleHorizontal_8bpc      | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingPaste                        | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingResampleVertical_8bpc        | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _int_free                           | libc.so.6     |
|                         | precompute_coeffs                   | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                        |                                    |               |
| RandomPosterize         | im_point_3x8_3x8                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __libc_calloc                       | libc.so.6     |
|                         | _int_free                           | libc.so.6     |
|                        |                                    |               |
| CenterCrop              | _int_free                           | libc.so.6     |
|                         | _Py_INCREF                          | python3.10    |
|                        |                                    |               |
| RandomGrayscale         | _aligned_contig_to_strided_size1    | _multiarray_umath.cpython-310-x86_64-linux-gnu.so |
|                         | lookdict_unicode_nodummy            | python3.10    |
|                         | __GI___libc_malloc                  | libc.so.6     |
|                         | _PyEval_EvalFrameDefault            | python3.10    |
|                         | stringlib_bytes_join                | python3.10    |
|                         | ImagingFill                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingUnpackRGB                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingRawEncode                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | arena_map_is_used                   | python3.10    |
|                         | __GI_                               | libc.so.6     |
|                         | rgb2l                               | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingRawDecode                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                         | __memset_avx2_erms                  | libc.so.6     |
|                         | munmap                              | libc.so.6     |
|                         | do_mkvalue                          | python3.10    |
|                        |                                    |               |
| Resize                  | ImagingResampleHorizontal_8bpc      | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | _int_free                           | libc.so.6     |
|                         | ImagingResampleVertical_8bpc        | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | normalize_coeffs_8bpc               | _imaging.cpython-310-x86_64-linux-gnu.so |
|                        |                                    |               |
| LinearTransformation    | gomp_simple_barrier_wait            | libgomp.so.1  |
|                         | do_spin                             | libgomp.so.1  |
|                        |                                    |               |
| RandomVerticalFlip      | _int_free                           | libc.so.6     |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                         | ImagingFlipTopBottom                | _imaging.cpython-310-x86_64-linux-gnu.so |
|                        |                                    |               |
| RandomAdjustSharpness   | __GI___libc_malloc                  | libc.so.6     |
|                         | ImagingBlend                        | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingFilter3x3                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | __libc_calloc                       | libc.so.6     |
|                        |                                    |               |
| RandomAffine            | _int_free                           | libc.so.6     |
|                         | ImagingFill                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingTransformAffine              | _imaging.cpython-310-x86_64-linux-gnu.so |
|                        |                                    |               |
| FiveCrop                | No Functions Captured               |               |
| GaussianBlur            | vgetargs1_impl                      | python3.10    |
|                         | gomp_simple_barrier_wait            | libgomp.so.1  |
|                         | PyObject_Malloc                     | python3.10    |
|                         | __close                             | libpthread.so.0 |
|                         | at::native::AVX2::round_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(char**long const*, long)#1}::operator().isra.0 | libtorch_cpu.so |
|                         | gomp_team_barrier_wait_end          | libgomp.so.1  |
|                         | std::_Function_handler<void (long, long, long), dnnl::impl::cpu::x64::jit_uni_reorder_t::omp_driver_3d(int, int, int, char const*, char*, float const*, float const*, int, int, int*) const::{lambda(longlong, long)#1}>::_M_invoke | libtorch_cpu.so |
|                         | PyArray_ToString                    | _multiarray_umath.cpython-310-x86_64-linux-gnu.so |
|                         | _PyObject_GetMethod                 | python3.10    |
|                         | std::_Function_handler<void (int, int), dnnl::impl::parallel_nd(long, long, long, std::function<void (long, long, long)> const&)::{lambda(intint)#1}>::_M_invoke | libtorch_cpu.so |
|                         | __memmove_avx_unaligned_erms        | libc.so.6     |
|                         | __memset_avx2_erms                  | libc.so.6     |
|                         | std::_Function_handler<void (int, int), dnnl::impl::parallel_nd(long, long, long, long, long, std::function<void (long, long, long, long, long)> const&)::{lambda(intint)#1}>::_M_invoke | libtorch_cpu.so |
|                         | dnnl::impl::cpu::x64::jit_uni_reorder_t::omp_driver_3d | libtorch_cpu.so |
|                         | dnnl::impl::cpu::x64::jit_uni_reorder_t::fill_curr_data_chunks | libtorch_cpu.so |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::TensorIteratorBase::loop_2d_from_1d<at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1}>(, signed char, at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1} const&)::{lambda(char**long const*, long, long)#1}> | libtorch_cpu.so |
|                         | __GI___libc_malloc                  | libc.so.6     |
|                         | _PyBytes_Resize                     | python3.10    |
|                         | stringlib_bytes_join                | python3.10    |
|                         | jit_uni_reorder_kernel_f32          | [Dynamic code] |
|                         | func@0xcc16b0                       | libtorch_cpu.so |
|                         | [OpenMP fork]                       | libgomp.so.1  |
|                         | dnnl::impl::cpu::x64::jit_uni_dw_convolution_fwd_t<(dnnl::impl::cpu::x64::cpu_isa_t)7, (dnnl_data_type_t)3, (dnnl_data_type_t)3>::execute_forward(dnnl::impl::exec_ctx_t const&) const::{lambda(intint)#1}::operator() | libtorch_cpu.so |
|                         | gomp_team_end                       | libgomp.so.1  |
|                         | __tls_get_addr                      | ld-linux-x86-64.so.2 |
|                         | syscall                             | libc.so.6     |
|                         | PyBuffer_FillInfo                   | python3.10    |
|                         | object_dealloc                      | python3.10    |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::TensorIteratorBase::loop_2d_from_1d<at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(char**long const*, long)#1}>(, signed char, at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(char**long const*, long)#1} const&)::{lambda(char**long const*, long, long)#1}> | libtorch_cpu.so |
|                         | _PyEval_EvalFrameDefault            | python3.10    |
|                         | jit_uni_dw_conv_fwd_kernel_f32      | [Dynamic code] |
|                         | dnnl::impl::itt::primitive_task_end | libtorch_cpu.so |
|                         | tuple_alloc                         | python3.10    |
|                         | ImagingUnpackRGB                    | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | cpu_relax                           | libgomp.so.1  |
|                         | [Unknown stack frame(s)]            | [Unknown]     |
|                         | do_mkvalue                          | python3.10    |
|                         | c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(float)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#4}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>)#2}>> | libtorch_cpu.so |
|                         | munmap                              | libc.so.6     |
|                         | do_spin                             | libgomp.so.1  |
|                         | _ZN2at8internal15invoke_parallelIZNS_6native12_GLOBAL__N_111cpu_paddingIfNS3_13ReflectionPadEEEvRKNS_6TensorES8_RNS3_13PaddingParamsEEUlllE1_EEvlllRKT_._omp_fn.0 | libtorch_cpu.so |
|                         | _PyDict_LoadGlobal                  | python3.10    |
|                         | OS_GetForksCount                    | libc-dynamic.so |
|                         | _Py_INCREF                          | python3.10    |
|                         | ImagingFill                         | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | ImagingPackRGB                      | _imaging.cpython-310-x86_64-linux-gnu.so |
|                         | PyObject_IsTrue                     | python3.10    |
|                         | __GI_                               | libc.so.6     |
|                         | __libc_open64                       | libpthread.so.0 |
|                        |                                    |               |
