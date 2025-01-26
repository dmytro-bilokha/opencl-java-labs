package com.dmytrobilokha.opencl.binding;

import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.Optional;

public final class MethodBinding {

    private static final String LIB_PATH_PROPERTY = "opencl.lib.path";
    private static final String DEFAULT_LIB_PATH = "/usr/local/lib/libOpenCL.so";

    static {
        var libPath = System.getProperty(LIB_PATH_PROPERTY, DEFAULT_LIB_PATH);
        try {
            System.load(libPath);
        } catch (RuntimeException e) {
            throw new IllegalStateException("Failed to load OpenCL lib from path: " + libPath, e);
        }
    }

    private static final Linker linker = Linker.nativeLinker();
    private static final SymbolLookup linkerLookup = linker.defaultLookup();
    private static final SymbolLookup classLoaderLookup = SymbolLookup.loaderLookup();

    private static final MemorySegment GET_PLATFORM_IDS_MEMSEG = lookupSymbol("clGetPlatformIDs");
    private static final FunctionDescriptor GET_PLATFORM_IDS_DESC = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);
    public static final MethodHandle GET_PLATFORM_IDS_HANDLE = linker.downcallHandle(
            GET_PLATFORM_IDS_MEMSEG, GET_PLATFORM_IDS_DESC);

    private static final MemorySegment GET_PLATFORM_INFO_MEMSEG =
            lookupSymbol("clGetPlatformInfo");
    private static final FunctionDescriptor GET_PLATFORM_INFO_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS,        // cl_platform_id platform
                    ValueLayout.JAVA_INT,       // cl_platform_info param_name
                    ValueLayout.JAVA_LONG,      // size_t param_value_size
                    ValueLayout.ADDRESS,        // void* param_value
                    ValueLayout.ADDRESS         // size_t* param_value_size_ret
            );
    public static final MethodHandle GET_PLATFORM_INFO_HANDLE =
            linker.downcallHandle(
                    GET_PLATFORM_INFO_MEMSEG,
                    GET_PLATFORM_INFO_DESC
            );

    private static final MemorySegment GET_DEVICE_IDS_MEMSEG = lookupSymbol("clGetDeviceIDs");
    private static final FunctionDescriptor GET_DEVICE_IDS_DESC = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,         // Return type: cl_int
            ValueLayout.ADDRESS,          // cl_platform_id platform
            ValueLayout.JAVA_LONG,        // cl_device_type device_type (unsigned long)
            ValueLayout.JAVA_INT,         // cl_uint num_entries
            ValueLayout.ADDRESS,          // cl_device_id* devices
            ValueLayout.ADDRESS           // cl_uint* num_devices
    );
    public static final MethodHandle GET_DEVICE_IDS_HANDLE = linker.downcallHandle(
            GET_DEVICE_IDS_MEMSEG, GET_DEVICE_IDS_DESC);

    private static final MemorySegment CREATE_CONTEXT_MEMSEG = lookupSymbol("clCreateContext");
    private static final FunctionDescriptor CREATE_CONTEXT_DESC = FunctionDescriptor.of(
            ValueLayout.ADDRESS,         // Return type: cl_context (void*)
            ValueLayout.ADDRESS,         // const cl_context_properties* properties
            ValueLayout.JAVA_INT,        // cl_uint num_devices
            ValueLayout.ADDRESS,         // const cl_device_id* devices
            ValueLayout.ADDRESS,         // void (CL_CALLBACK *pfn_notify)(...)
            ValueLayout.ADDRESS,         // void* user_data
            ValueLayout.ADDRESS          // cl_int* errcode_ret
    );
    public static final MethodHandle CREATE_CONTEXT_HANDLE = linker.downcallHandle(
            CREATE_CONTEXT_MEMSEG, CREATE_CONTEXT_DESC);

    private static final MemorySegment RELEASE_CONTEXT_MEMSEG = lookupSymbol("clReleaseContext");
    private static final FunctionDescriptor RELEASE_CONTEXT_DESC = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,         // Return type: cl_int
            ValueLayout.ADDRESS           // cl_context context
    );
    public static final MethodHandle RELEASE_CONTEXT_HANDLE = linker.downcallHandle(
            RELEASE_CONTEXT_MEMSEG, RELEASE_CONTEXT_DESC);

    private static final MemorySegment GET_DEVICE_INFO_MEMSEG = lookupSymbol("clGetDeviceInfo");
    private static final FunctionDescriptor GET_DEVICE_INFO_DESC = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,         // Return type: cl_int
            ValueLayout.ADDRESS,          // cl_device_id device
            ValueLayout.JAVA_LONG,        // cl_device_info param_name
            ValueLayout.JAVA_LONG,        // size_t param_value_size
            ValueLayout.ADDRESS,          // void* param_value
            ValueLayout.ADDRESS           // size_t* param_value_size_ret
    );
    public static final MethodHandle GET_DEVICE_INFO_HANDLE = linker.downcallHandle(
            GET_DEVICE_INFO_MEMSEG, GET_DEVICE_INFO_DESC);

    private static final MemorySegment CREATE_BUFFER_MEMSEG = lookupSymbol("clCreateBuffer");
    private static final FunctionDescriptor CREATE_BUFFER_DESC = FunctionDescriptor.of(
            ValueLayout.ADDRESS,            // Return type: cl_mem (void*)
            ValueLayout.ADDRESS,            // cl_context context
            ValueLayout.JAVA_LONG,          // cl_mem_flags flags
            ValueLayout.JAVA_LONG,          // size_t size
            ValueLayout.ADDRESS,            // void* host_ptr
            ValueLayout.ADDRESS             // cl_int* errcode_ret
    );
    public static final MethodHandle CREATE_BUFFER_HANDLE = linker.downcallHandle(
            CREATE_BUFFER_MEMSEG, CREATE_BUFFER_DESC);

    private static final MemorySegment CREATE_COMMAND_QUEUE_WITH_PROPERTIES_MEMSEG =
            lookupSymbol("clCreateCommandQueueWithProperties");
    private static final FunctionDescriptor CREATE_COMMAND_QUEUE_WITH_PROPERTIES_DESC =
            FunctionDescriptor.of(
                    ValueLayout.ADDRESS,       // Return type: cl_command_queue
                    ValueLayout.ADDRESS,       // cl_context context
                    ValueLayout.ADDRESS,       // cl_device_id device
                    ValueLayout.ADDRESS,       // const cl_queue_properties* properties
                    ValueLayout.ADDRESS        // cl_int* errcode_ret
            );
    public static final MethodHandle CREATE_COMMAND_QUEUE_WITH_PROPERTIES_HANDLE =
            linker.downcallHandle(
                    CREATE_COMMAND_QUEUE_WITH_PROPERTIES_MEMSEG,
                    CREATE_COMMAND_QUEUE_WITH_PROPERTIES_DESC
            );

    private static final MemorySegment CREATE_PROGRAM_WITH_SOURCE_MEMSEG =
            lookupSymbol("clCreateProgramWithSource");
    private static final FunctionDescriptor CREATE_PROGRAM_WITH_SOURCE_DESC =
            FunctionDescriptor.of(
                    ValueLayout.ADDRESS,       // Return type: cl_program
                    ValueLayout.ADDRESS,       // cl_context context
                    ValueLayout.JAVA_INT,      // cl_uint count
                    ValueLayout.ADDRESS,       // const char** strings
                    ValueLayout.ADDRESS,       // const size_t* lengths
                    ValueLayout.ADDRESS        // cl_int* errcode_ret
            );
    public static final MethodHandle CREATE_PROGRAM_WITH_SOURCE_HANDLE =
            linker.downcallHandle(
                    CREATE_PROGRAM_WITH_SOURCE_MEMSEG,
                    CREATE_PROGRAM_WITH_SOURCE_DESC
            );

    private static final MemorySegment BUILD_PROGRAM_MEMSEG =
            lookupSymbol("clBuildProgram");
    private static final FunctionDescriptor BUILD_PROGRAM_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,  // Return type: cl_int
                    ValueLayout.ADDRESS,   // cl_program program
                    ValueLayout.JAVA_INT,  // cl_uint num_devices
                    ValueLayout.ADDRESS,   // const cl_device_id* device_list
                    ValueLayout.ADDRESS,   // const char* options
                    ValueLayout.ADDRESS,   // void (*pfn_notify)(cl_program, void*)
                    ValueLayout.ADDRESS    // void* user_data
            );
    public static final MethodHandle BUILD_PROGRAM_HANDLE =
            linker.downcallHandle(
                    BUILD_PROGRAM_MEMSEG,
                    BUILD_PROGRAM_DESC
            );

    private static final MemorySegment GET_PROGRAM_BUILD_INFO_MEMSEG =
            lookupSymbol("clGetProgramBuildInfo");
    private static final FunctionDescriptor GET_PROGRAM_BUILD_INFO_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS,        // cl_program program
                    ValueLayout.ADDRESS,        // cl_device_id device
                    ValueLayout.JAVA_INT,       // cl_program_build_info param_name
                    ValueLayout.JAVA_LONG,      // size_t param_value_size
                    ValueLayout.ADDRESS,        // void* param_value
                    ValueLayout.ADDRESS         // size_t* param_value_size_ret
            );
    public static final MethodHandle GET_PROGRAM_BUILD_INFO_HANDLE =
            linker.downcallHandle(
                    GET_PROGRAM_BUILD_INFO_MEMSEG,
                    GET_PROGRAM_BUILD_INFO_DESC
            );

    private static final MemorySegment CREATE_KERNEL_MEMSEG =
            lookupSymbol("clCreateKernel");
    private static final FunctionDescriptor CREATE_KERNEL_DESC =
            FunctionDescriptor.of(
                    ValueLayout.ADDRESS,       // Return type: cl_kernel
                    ValueLayout.ADDRESS,       // cl_program program
                    ValueLayout.ADDRESS,       // const char* kernel_name
                    ValueLayout.ADDRESS        // cl_int* errcode_ret
            );
    public static final MethodHandle CREATE_KERNEL_HANDLE =
            linker.downcallHandle(
                    CREATE_KERNEL_MEMSEG,
                    CREATE_KERNEL_DESC
            );

    private static final MemorySegment SET_KERNEL_ARG_MEMSEG =
            lookupSymbol("clSetKernelArg");
    private static final FunctionDescriptor SET_KERNEL_ARG_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS,        // cl_kernel kernel
                    ValueLayout.JAVA_INT,       // cl_uint arg_index
                    ValueLayout.JAVA_LONG,      // size_t arg_size
                    ValueLayout.ADDRESS         // const void* arg_value
            );
    public static final MethodHandle SET_KERNEL_ARG_HANDLE =
            linker.downcallHandle(
                    SET_KERNEL_ARG_MEMSEG,
                    SET_KERNEL_ARG_DESC
            );

    private static final MemorySegment ENQUEUE_ND_RANGE_KERNEL_MEMSEG =
            lookupSymbol("clEnqueueNDRangeKernel");
    private static final FunctionDescriptor ENQUEUE_ND_RANGE_KERNEL_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS,        // cl_command_queue command_queue
                    ValueLayout.ADDRESS,        // cl_kernel kernel
                    ValueLayout.JAVA_INT,       // cl_uint work_dim
                    ValueLayout.ADDRESS,        // const size_t* global_work_offset
                    ValueLayout.ADDRESS,        // const size_t* global_work_size
                    ValueLayout.ADDRESS,        // const size_t* local_work_size
                    ValueLayout.JAVA_INT,       // cl_uint num_events_in_wait_list
                    ValueLayout.ADDRESS,        // const cl_event* event_wait_list
                    ValueLayout.ADDRESS         // cl_event* event
            );
    public static final MethodHandle ENQUEUE_ND_RANGE_KERNEL_HANDLE =
            linker.downcallHandle(
                    ENQUEUE_ND_RANGE_KERNEL_MEMSEG,
                    ENQUEUE_ND_RANGE_KERNEL_DESC
            );

    private static final MemorySegment FINISH_MEMSEG =
            lookupSymbol("clFinish");
    private static final FunctionDescriptor FINISH_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,      // Return type: cl_int
                    ValueLayout.ADDRESS        // cl_command_queue command_queue
            );
    public static final MethodHandle FINISH_HANDLE =
            linker.downcallHandle(
                    FINISH_MEMSEG,
                    FINISH_DESC
            );

    private static final MemorySegment ENQUEUE_READ_BUFFER_MEMSEG =
            lookupSymbol("clEnqueueReadBuffer");
    private static final FunctionDescriptor ENQUEUE_READ_BUFFER_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS,        // cl_command_queue command_queue
                    ValueLayout.ADDRESS,        // cl_mem buffer
                    ValueLayout.JAVA_INT,       // cl_bool blocking_read
                    ValueLayout.JAVA_LONG,      // size_t offset
                    ValueLayout.JAVA_LONG,      // size_t size
                    ValueLayout.ADDRESS,        // void* ptr
                    ValueLayout.JAVA_INT,       // cl_uint num_events_in_wait_list
                    ValueLayout.ADDRESS,        // const cl_event* event_wait_list
                    ValueLayout.ADDRESS         // cl_event* event
            );
    public static final MethodHandle ENQUEUE_READ_BUFFER_HANDLE =
            linker.downcallHandle(
                    ENQUEUE_READ_BUFFER_MEMSEG,
                    ENQUEUE_READ_BUFFER_DESC
            );

    private static final MemorySegment ENQUEUE_WRITE_BUFFER_MEMSEG =
            lookupSymbol("clEnqueueWriteBuffer");
    private static final FunctionDescriptor ENQUEUE_WRITE_BUFFER_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS,        // cl_command_queue command_queue
                    ValueLayout.ADDRESS,        // cl_mem buffer
                    ValueLayout.JAVA_INT,       // cl_bool blocking_write
                    ValueLayout.JAVA_LONG,      // size_t offset
                    ValueLayout.JAVA_LONG,      // size_t size
                    ValueLayout.ADDRESS,        // void* ptr
                    ValueLayout.JAVA_INT,       // cl_uint num_events_in_wait_list
                    ValueLayout.ADDRESS,        // const cl_event* event_wait_list
                    ValueLayout.ADDRESS         // cl_event* event
            );
    public static final MethodHandle ENQUEUE_WRITE_BUFFER_HANDLE =
            linker.downcallHandle(
                    ENQUEUE_WRITE_BUFFER_MEMSEG,
                    ENQUEUE_WRITE_BUFFER_DESC
            );

    private static final MemorySegment RELEASE_MEM_OBJECT_MEMSEG =
            lookupSymbol("clReleaseMemObject");
    private static final FunctionDescriptor RELEASE_MEM_OBJECT_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS         // cl_mem memobj
            );
    public static final MethodHandle RELEASE_MEM_OBJECT_HANDLE =
            linker.downcallHandle(
                    RELEASE_MEM_OBJECT_MEMSEG,
                    RELEASE_MEM_OBJECT_DESC
            );

    private static final MemorySegment RELEASE_KERNEL_MEMSEG =
            lookupSymbol("clReleaseKernel");
    private static final FunctionDescriptor RELEASE_KERNEL_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS         // cl_kernel kernel
            );
    public static final MethodHandle RELEASE_KERNEL_HANDLE =
            linker.downcallHandle(
                    RELEASE_KERNEL_MEMSEG,
                    RELEASE_KERNEL_DESC
            );

    private static final MemorySegment RELEASE_PROGRAM_MEMSEG =
            lookupSymbol("clReleaseProgram");
    private static final FunctionDescriptor RELEASE_PROGRAM_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS         // cl_program program
            );
    public static final MethodHandle RELEASE_PROGRAM_HANDLE =
            linker.downcallHandle(
                    RELEASE_PROGRAM_MEMSEG,
                    RELEASE_PROGRAM_DESC
            );

    private static final MemorySegment RELEASE_COMMAND_QUEUE_MEMSEG =
            lookupSymbol("clReleaseCommandQueue");
    private static final FunctionDescriptor RELEASE_COMMAND_QUEUE_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,       // Return type: cl_int
                    ValueLayout.ADDRESS         // cl_command_queue command_queue
            );
    public static final MethodHandle RELEASE_COMMAND_QUEUE_HANDLE =
            linker.downcallHandle(
                    RELEASE_COMMAND_QUEUE_MEMSEG,
                    RELEASE_COMMAND_QUEUE_DESC
            );

    private static final MemorySegment GET_COMMAND_QUEUE_INFO_MEMSEG =
            lookupSymbol("clGetCommandQueueInfo");

    private static final FunctionDescriptor GET_COMMAND_QUEUE_INFO_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,        // Return type: cl_int
                    ValueLayout.ADDRESS,         // cl_command_queue command_queue
                    ValueLayout.JAVA_INT,        // cl_command_queue_info param_name
                    ValueLayout.JAVA_LONG,       // size_t param_value_size
                    ValueLayout.ADDRESS,         // void* param_value
                    ValueLayout.ADDRESS          // size_t* param_value_size_ret
            );

    public static final MethodHandle GET_COMMAND_QUEUE_INFO_HANDLE =
            linker.downcallHandle(
                    GET_COMMAND_QUEUE_INFO_MEMSEG,
                    GET_COMMAND_QUEUE_INFO_DESC
            );

    private MethodBinding() {
        // no instantiation
    }

    private static MemorySegment lookupSymbol(String symbol) {
        Optional<MemorySegment> lookupResult = classLoaderLookup.find(symbol);
        if (lookupResult.isPresent()) {
            return lookupResult.get();
        }
        lookupResult = linkerLookup.find(symbol);
        if (lookupResult.isPresent()) {
            return lookupResult.get();
        }
        throw new OpenClRuntimeException("Lookup failed for symbol: " + symbol);
    }

    public static MemorySegment invokeMemSegClMethod(
            MemorySegment errorCodeMemSeg, MethodHandle methodHandle, Object... arguments) {
        MemorySegment returnValue;
        try {
            returnValue = (MemorySegment) methodHandle.invokeWithArguments(arguments);
        } catch (Throwable e) {
            throw new OpenClRuntimeException("Failed to call " + methodHandle + " with parameters: "
                    + Arrays.toString(arguments), e);
        }
        int errorCode = errorCodeMemSeg.get(ValueLayout.JAVA_INT, 0);
        if (!ReturnValue.CL_SUCCESS.matches(errorCode)) {
            throw new OpenClRuntimeException(
                    "Error " + ReturnValue.convertToString(errorCode) + " while calling " + methodHandle);
        }
        return returnValue;
    }

    public static void invokeClMethod(MethodHandle methodHandle, Object... arguments) {
        int returnValue;
        try {
            returnValue = (int) methodHandle.invokeWithArguments(arguments);
        } catch (Throwable e) {
            throw new OpenClRuntimeException("Failed to call " + methodHandle + " with parameters: "
                    + Arrays.toString(arguments), e);
        }
        if (!ReturnValue.CL_SUCCESS.matches(returnValue)) {
            throw new OpenClRuntimeException(
                    "Error " + ReturnValue.convertToString(returnValue) + " while calling " + methodHandle
                + " with parameters: " + Arrays.toString(arguments));
        }
    }
}
