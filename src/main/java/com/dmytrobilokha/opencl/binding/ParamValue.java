package com.dmytrobilokha.opencl.binding;

public final class ParamValue {

    public static final int CL_DEVICE_TYPE_DEFAULT = (1 << 0);
    public static final int CL_DEVICE_TYPE_CPU = (1 << 1);
    public static final int CL_DEVICE_TYPE_GPU = (1 << 2);
    public static final int CL_DEVICE_TYPE_ACCELERATOR = (1 << 3);
    public static final int CL_DEVICE_TYPE_CUSTOM = (1 << 4);
    public static final int CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;

    /* cl_mem_flags and cl_svm_mem_flags - bitfield */
    public static final long CL_MEM_READ_WRITE = (1 << 0);
    public static final long CL_MEM_WRITE_ONLY = (1 << 1);
    public static final long CL_MEM_READ_ONLY = (1 << 2);
    public static final long CL_MEM_USE_HOST_PTR = (1 << 3);
    public static final long CL_MEM_ALLOC_HOST_PTR = (1 << 4);
    public static final long CL_MEM_COPY_HOST_PTR = (1 << 5);
    public static final long CL_MEM_HOST_WRITE_ONLY = (1 << 7);
    public static final long CL_MEM_HOST_READ_ONLY = (1 << 8);
    public static final long CL_MEM_HOST_NO_ACCESS = (1 << 9);
    public static final long CL_MEM_SVM_FINE_GRAIN_BUFFER = (1 << 10) /* used by cl_svm_mem_flags only */;
    public static final long CL_MEM_SVM_ATOMICS = (1 << 11) /* used by cl_svm_mem_flags only */;
    public static final long CL_MEM_KERNEL_READ_AND_WRITE = (1 << 12);

    /* cl_bool */
    public static final int CL_FALSE = 0;
    public static final int CL_TRUE = 1;
    public static final int CL_BLOCKING = CL_TRUE;
    public static final int CL_NON_BLOCKING = CL_FALSE;

    /* cl_platform_info */
    /* Returns the profile string supported by the platform. Expected type: String. */
    public static final int CL_PLATFORM_PROFILE = 0x0900;
    /* Returns the name string of the platform vendor. Expected type: String. */
    public static final int CL_PLATFORM_VERSION = 0x0901;
    /* Returns the platform name. Expected type: String. */
    public static final int CL_PLATFORM_NAME = 0x0902;
    /* Returns the platform vendor string. Expected type: String. */
    public static final int CL_PLATFORM_VENDOR = 0x0903;
    /* Returns space-separated extension names supported by the platform. Expected type: String. */
    public static final int CL_PLATFORM_EXTENSIONS = 0x0904;
    /* Returns the numeric version of the OpenCL implementation. Expected type: cl_version (encoded version number). */
    public static final int CL_PLATFORM_NUMERIC_VERSION = 0x0906;
    /* Returns supported OpenCL extension names along with their versions. Expected type: Array of cl_name_version. */
    public static final int CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907;
    /* Returns the ICD loader suffix string for the platform, if present. Expected type: String. */
    public static final int CL_PLATFORM_ICD_SUFFIX_KHR = 0x0920;

    /* cl_device_info */
    /* Returns the type of the device (e.g., CPU, GPU). Expected Java type: long. */
    public static final long CL_DEVICE_TYPE = 0x1000;
    /* Returns the vendor ID of the device. Expected Java type: long. */
    public static final long CL_DEVICE_VENDOR_ID = 0x1001;
    /* Returns the number of compute units. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
    /* Returns the maximum number of dimensions supported for work items. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003;
    /* Returns the maximum size of a work group. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
    /* Returns the maximum sizes of work items in each dimension. Expected Java type: long array. */
    public static final long CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005;
    /* Returns the preferred vector width for char. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006;
    /* Returns the preferred vector width for short. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007;
    /* Returns the preferred vector width for int. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008;
    /* Returns the preferred vector width for long. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009;
    /* Returns the preferred vector width for float. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A;
    /* Returns the preferred vector width for double. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100B;
    /* Returns the device's maximum clock frequency in MHz. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C;
    /* Returns the address bits of the device. Expected Java type: long. */
    public static final long CL_DEVICE_ADDRESS_BITS = 0x100D;
    /* Returns the maximum number of read-only image arguments. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100E;
    /* Returns the maximum number of write-only image arguments. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100F;
    /* Returns the maximum size of memory object allocation in bytes. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010;
    /* Returns the maximum width of 2D images. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011;
    /* Returns the maximum height of 2D images. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012;
    /* Returns the maximum width of 3D images. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013;
    /* Returns the maximum height of 3D images. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014;
    /* Returns the maximum depth of 3D images. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015;
    /* Returns whether images are supported. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_IMAGE_SUPPORT = 0x1016;
    /* Returns the maximum size in bytes of a parameter passed to a kernel. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017;
    /* Returns the maximum number of samplers. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_SAMPLERS = 0x1018;
    /* Returns the alignment requirement for base addresses. Expected Java type: long. */
    public static final long CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019;
    /* Returns the smallest alignment size for a data type. Expected Java type: long. */
    public static final long CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A;
    /* Returns the single-precision floating-point configuration. Expected Java type: long. */
    public static final long CL_DEVICE_SINGLE_FP_CONFIG = 0x101B;
    /* Returns the type of global memory cache. Expected Java type: long. */
    public static final long CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C;
    /* Returns the size of global memory cache lines in bytes. Expected Java type: long. */
    public static final long CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D;
    /* Returns the size of global memory cache in bytes. Expected Java type: long. */
    public static final long CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E;
    /* Returns the size of global memory in bytes. Expected Java type: long. */
    public static final long CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
    /* Returns the maximum size in bytes of a constant buffer. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020;
    /* Returns the maximum number of constant buffer arguments. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021;
    /* Returns the type of local memory. Expected Java type: long. */
    public static final long CL_DEVICE_LOCAL_MEM_TYPE = 0x1022;
    /* Returns the size of local memory in bytes. Expected Java type: long. */
    public static final long CL_DEVICE_LOCAL_MEM_SIZE = 0x1023;
    /* Returns whether error correction is supported. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024;
    /* Returns the profiling timer resolution in nanoseconds. Expected Java type: long. */
    public static final long CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025;
    /* Returns whether the device is little-endian. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_ENDIAN_LITTLE = 0x1026;
    /* Returns whether the device is available. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_AVAILABLE = 0x1027;
    /* Returns whether the OpenCL compiler is available. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_COMPILER_AVAILABLE = 0x1028;
    /* Returns the execution capabilities of the device. Expected Java type: long. */
    public static final long CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029;
    /* Returns the on-host queue properties (deprecated). Expected Java type: long. */
    public static final long CL_DEVICE_QUEUE_PROPERTIES = 0x102A;
    /* Returns the name of the device. Expected Java type: String. */
    public static final long CL_DEVICE_NAME = 0x102B;
    /* Returns the vendor of the device. Expected Java type: String. */
    public static final long CL_DEVICE_VENDOR = 0x102C;
    /* Returns the driver version. Expected Java type: String. */
    public static final long CL_DRIVER_VERSION = 0x102D;
    /* Returns the device's OpenCL profile. Expected Java type: String. */
    public static final long CL_DEVICE_PROFILE = 0x102E;
    /* Returns the OpenCL version supported by the device. Expected Java type: String. */
    public static final long CL_DEVICE_VERSION = 0x102F;
    /* Returns the extensions supported by the device. Expected Java type: String. */
    public static final long CL_DEVICE_EXTENSIONS = 0x1030;
    /* Returns the platform associated with the device. Expected Java type: MemorySegment (long). */
    public static final long CL_DEVICE_PLATFORM = 0x1031;
    /* Returns the double-precision floating-point configuration. Expected Java type: long. */
    public static final long CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032;
    /* Returns the preferred vector width for half. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034;
    /* Returns whether the device shares memory with the host (deprecated). Expected Java type: boolean (long). */
    public static final long CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035;
    /* Returns the native vector width for char. Expected Java type: long. */
    public static final long CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036;
    /* Returns the native vector width for short. Expected Java type: long. */
    public static final long CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037;
    /* Returns the native vector width for int. Expected Java type: long. */
    public static final long CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038;
    /* Returns the native vector width for long. Expected Java type: long. */
    public static final long CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039;
    /* Returns the native vector width for float. Expected Java type: long. */
    public static final long CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A;
    /* Returns the native vector width for double. Expected Java type: long. */
    public static final long CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B;
    /* Returns the native vector width for half. Expected Java type: long. */
    public static final long CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C;
    /* Returns the OpenCL C version supported by the device. Expected Java type: String. */
    public static final long CL_DEVICE_OPENCL_C_VERSION = 0x103D;
    /* Returns whether the device linker is available. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_LINKER_AVAILABLE = 0x103E;
    /* Returns the built-in kernels supported by the device. Expected Java type: String. */
    public static final long CL_DEVICE_BUILT_IN_KERNELS = 0x103F;
    /* Returns the maximum size for buffers used for image data. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040;
    /* Returns the maximum size for image arrays. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041;
    /* Returns the parent device, if this device is a sub-device. Expected Java type: MemorySegment (long). */
    public static final long CL_DEVICE_PARENT_DEVICE = 0x1042;
    /* Returns the maximum number of sub-devices. Expected Java type: long. */
    public static final long CL_DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043;
    /* Returns the properties for device partitioning. Expected Java type: long array. */
    public static final long CL_DEVICE_PARTITION_PROPERTIES = 0x1044;
    /* Returns the supported partition affinity domains. Expected Java type: long. */
    public static final long CL_DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045;
    /* Returns the partition type for sub-devices. Expected Java type: long array. */
    public static final long CL_DEVICE_PARTITION_TYPE = 0x1046;
    /* Returns the reference count of the device. Expected Java type: long. */
    public static final long CL_DEVICE_REFERENCE_COUNT = 0x1047;
    /* Returns whether interop with user synchronization is preferred. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048;
    /* Returns the size of the printf buffer. Expected Java type: long. */
    public static final long CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049;
    /* Returns the pitch alignment of images. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104A;
    /* Returns the base address alignment for images. Expected Java type: long. */
    public static final long CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104B;
    /* Returns the maximum number of simultaneous read-write images. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS = 0x104C;
    /* Returns the maximum size of global variables. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE = 0x104D;
    /* Returns properties of on-device command queues. Expected Java type: long. */
    public static final long CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES = 0x104E;
    /* Returns the preferred size for on-device queues. Expected Java type: long. */
    public static final long CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE = 0x104F;
    /* Returns the maximum size for on-device queues. Expected Java type: long. */
    public static final long CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = 0x1050;
    /* Returns the maximum number of on-device queues. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_ON_DEVICE_QUEUES = 0x1051;
    /* Returns the maximum number of on-device events. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_ON_DEVICE_EVENTS = 0x1052;
    /* Returns the SVM capabilities of the device. Expected Java type: long. */
    public static final long CL_DEVICE_SVM_CAPABILITIES = 0x1053;
    /* Returns the preferred total size of global variables. Expected Java type: long. */
    public static final long CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = 0x1054;
    /* Returns the maximum number of pipes supported. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_PIPE_ARGS = 0x1055;
    /* Returns the maximum active reservations for a pipe. Expected Java type: long. */
    public static final long CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056;
    /* Returns the maximum packet size for a pipe. Expected Java type: long. */
    public static final long CL_DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057;
    /* Returns the preferred platform atomic alignment. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058;
    /* Returns the preferred global atomic alignment. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059;
    /* Returns the preferred local atomic alignment. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105A;
    /* Returns the intermediate language version supported. Expected Java type: String. */
    public static final long CL_DEVICE_IL_VERSION = 0x105B;
    /* Returns the maximum number of sub-groups. Expected Java type: long. */
    public static final long CL_DEVICE_MAX_NUM_SUB_GROUPS = 0x105C;
    /* Returns whether sub-group independent forward progress is supported. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105D;
    /* Returns the OpenCL version in numeric format. Expected Java type: long. */
    public static final long CL_DEVICE_NUMERIC_VERSION = 0x105E;
    /* Returns supported device extensions with versions. Expected Java type: Struct/long array. */
    public static final long CL_DEVICE_EXTENSIONS_WITH_VERSION = 0x1060;
    /* Returns supported intermediate languages with versions. Expected Java type: Struct/long array. */
    public static final long CL_DEVICE_ILS_WITH_VERSION = 0x1061;
    /* Returns built-in kernels with versions. Expected Java type: Struct/long array. */
    public static final long CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION = 0x1062;
    /* Returns atomic memory capabilities. Expected Java type: long. */
    public static final long CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES = 0x1063;
    /* Returns atomic fence capabilities. Expected Java type: long. */
    public static final long CL_DEVICE_ATOMIC_FENCE_CAPABILITIES = 0x1064;
    /* Returns whether non-uniform work-group support is available. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT = 0x1065;
    /* Returns the OpenCL C all versions supported. Expected Java type: Struct/long array. */
    public static final long CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066;
    /* Returns the preferred work-group size multiple for kernels executed on the device. Expected Java type: long. */
    public static final long CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x1067;
    /* Returns whether work-group collective functions are supported. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = 0x1068;
    /* Returns whether generic address space is supported. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT = 0x1069;
    /* Returns the OpenCL C features supported by the device. Expected Java type: Struct/long array. */
    public static final long CL_DEVICE_OPENCL_C_FEATURES = 0x106F;
    /* Returns the capabilities of the device enqueue feature. Expected Java type: long. */
    public static final long CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES = 0x1070;
    /* Returns whether pipes are supported by the device. Expected Java type: boolean (long). */
    public static final long CL_DEVICE_PIPE_SUPPORT = 0x1071;
    /* Returns the latest OpenCL conformance test suite version passed. Expected Java type: String. */
    public static final long CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED = 0x1072;
    /* Returns the capabilities of command buffers supported by the device (KHR extension). Expected Java type: long. */
    public static final long CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR = 0x12A9;
    /* Returns the required queue properties for using command buffers on the device (KHR extension). Expected Java type: long. */
    public static final long CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR = 0x12AA;

    /* cl_command_queue_info */
    public static final long CL_QUEUE_CONTEXT = 0x1090;
    public static final long CL_QUEUE_DEVICE = 0x1091;
    public static final long CL_QUEUE_REFERENCE_COUNT = 0x1092;
    public static final long CL_QUEUE_PROPERTIES = 0x1093;
    public static final long CL_QUEUE_SIZE = 0x1094;
    public static final long CL_QUEUE_DEVICE_DEFAULT = 0x1095;
    public static final long CL_QUEUE_PROPERTIES_ARRAY = 0x1098;

    /* cl_command_queue_properties - bitfield */
    public static final long CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = (1 << 0);
    public static final long CL_QUEUE_PROFILING_ENABLE = (1 << 1);
    public static final long CL_QUEUE_ON_DEVICE = (1 << 2);
    public static final long CL_QUEUE_ON_DEVICE_DEFAULT = (1 << 3);

    /* cl_profiling_info */
    public static final int CL_PROFILING_COMMAND_QUEUED = 0x1280;
    public static final int CL_PROFILING_COMMAND_SUBMIT = 0x1281;
    public static final int CL_PROFILING_COMMAND_START = 0x1282;
    public static final int CL_PROFILING_COMMAND_END = 0x1283;
    public static final int CL_PROFILING_COMMAND_COMPLETE = 0x1284;

    private ParamValue() {
        // no instance
    }
}
