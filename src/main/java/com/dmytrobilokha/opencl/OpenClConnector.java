package com.dmytrobilokha.opencl;

import com.dmytrobilokha.opencl.binding.ParamValue;
import com.dmytrobilokha.opencl.binding.ReturnValue;
import com.dmytrobilokha.opencl.binding.MethodBinding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;

public class OpenClConnector {

    private static final long DEVICE_NAME_LIMIT = 250;

    private final Arena arena;

    public OpenClConnector(Arena arena) {
        this.arena = arena;
    }

    public MemorySegment getDefaultPlatformDeviceIds() {
        var numPlatformsMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        invokeClMethod(MethodBinding.GET_PLATFORM_IDS_HANDLE, 0, MemorySegment.NULL, numPlatformsMemSeg);
        int numPlatforms = numPlatformsMemSeg.get(ValueLayout.JAVA_INT, 0);
        if (numPlatforms < 1) {
            throw new IllegalStateException("No OpenCL platforms found, unable to continue");
        }
        var platformIdsMemSeg = arena.allocate(ValueLayout.ADDRESS, numPlatforms);
        invokeClMethod(
                MethodBinding.GET_PLATFORM_IDS_HANDLE,
                numPlatforms,
                platformIdsMemSeg,
                MemorySegment.NULL);
        var defaultPlatform = platformIdsMemSeg.getAtIndex(ValueLayout.ADDRESS, 0);
        var numDevicesMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        invokeClMethod(MethodBinding.GET_DEVICE_IDS_HANDLE,
                defaultPlatform,
                ParamValue.CL_DEVICE_TYPE_ALL,
                0,
                MemorySegment.NULL,
                numDevicesMemSeg);
        int numDevices = numDevicesMemSeg.get(ValueLayout.JAVA_INT, 0);
        if (numDevices < 1) {
            throw new IllegalStateException("No OpenCL devices found for the default platform, unable to continue");
        }
        var deviceIdsMemSeg = arena.allocate(ValueLayout.ADDRESS, numDevices);
        invokeClMethod(
                MethodBinding.GET_DEVICE_IDS_HANDLE,
                defaultPlatform,
                ParamValue.CL_DEVICE_TYPE_ALL,
                numDevices,
                deviceIdsMemSeg,
                MemorySegment.NULL);
        return deviceIdsMemSeg;
    }

    public String getDeviceName(MemorySegment deviceIdMemSeg) {
        var deviceNameMemSeg = arena.allocate(DEVICE_NAME_LIMIT);
        invokeClMethod(
                MethodBinding.GET_DEVICE_INFO_HANDLE,
                deviceIdMemSeg,
                ParamValue.CL_DEVICE_NAME,
                DEVICE_NAME_LIMIT,
                deviceNameMemSeg,
                MemorySegment.NULL
        );
        return deviceNameMemSeg.getString(0);
    }

    public MemorySegment createContext(MemorySegment deviceIdsMemSeg) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        return invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_CONTEXT_HANDLE,
                MemorySegment.NULL,
                1,
                deviceIdsMemSeg,
                MemorySegment.NULL,
                MemorySegment.NULL,
                errorCodeMemSeg);
    }

    public void releaseContext(MemorySegment contextMemSeg) {
        invokeClMethod(MethodBinding.RELEASE_CONTEXT_HANDLE, contextMemSeg);
    }

    public MemorySegment createInputBufferOfFloats(MemorySegment contextMemSeg, float[] inputValues) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        var inputMemSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, inputValues);
        return invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_BUFFER_HANDLE,
                contextMemSeg,
                ParamValue.CL_MEM_READ_ONLY | ParamValue.CL_MEM_COPY_HOST_PTR,
                inputMemSeg.byteSize(),
                inputMemSeg,
                errorCodeMemSeg);
    }

    public MemorySegment createOutputBuffer(MemorySegment contextMemSeg, long byteSize) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        return invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_BUFFER_HANDLE,
                contextMemSeg,
                ParamValue.CL_MEM_WRITE_ONLY,
                byteSize,
                MemorySegment.NULL,
                errorCodeMemSeg);
    }

    public MemorySegment createCommandQueue(MemorySegment contextMemSeg, MemorySegment deviceIdMemSeg) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        return invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_COMMAND_QUEUE_WITH_PROPERTIES_HANDLE,
                contextMemSeg,
                deviceIdMemSeg,
                MemorySegment.NULL, //No queue properties for now
                errorCodeMemSeg);
    }

    public MemorySegment createProgram(MemorySegment contextMemSeg, String sourceCode) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        var sourceCodeMemSeg = arena.allocateFrom(sourceCode);
        var pointerToSourceCodeMemSeg = arena.allocate(ValueLayout.ADDRESS);
        pointerToSourceCodeMemSeg.set(ValueLayout.ADDRESS, 0, sourceCodeMemSeg);
        return invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_PROGRAM_WITH_SOURCE_HANDLE,
                contextMemSeg,
                1,
                pointerToSourceCodeMemSeg,
                MemorySegment.NULL,
                errorCodeMemSeg);
    }

    public void buildProgram(MemorySegment programMemSeg, MemorySegment deviceIdsMemSeg) {
        invokeClMethod(
                MethodBinding.BUILD_PROGRAM_HANDLE,
                programMemSeg,
                1,
                deviceIdsMemSeg,
                MemorySegment.NULL,
                MemorySegment.NULL,
                MemorySegment.NULL
        );
    }

    public MemorySegment createKernel(MemorySegment programMemSeg, String entryMethodName) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        return invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_KERNEL_HANDLE,
                programMemSeg,
                arena.allocateFrom(entryMethodName),
                errorCodeMemSeg
        );
    }

    public void setKernelArgument(MemorySegment kernelMemSeg, int argumentIndex, MemorySegment argumentMemSeg) {
        var argumentPointerMemSeg = arena.allocate(ValueLayout.ADDRESS);
        argumentPointerMemSeg.set(ValueLayout.ADDRESS, 0, argumentMemSeg);
        invokeClMethod(
                MethodBinding.SET_KERNEL_ARG_HANDLE,
                kernelMemSeg,
                argumentIndex,
                argumentPointerMemSeg.byteSize(),
                argumentPointerMemSeg);
    }

    public void enqueueNdRangeKernel(MemorySegment commandQueueMemSeg, MemorySegment kernelMemSeg, long workSize) {
        var workSizeMemSeg = arena.allocateFrom(ValueLayout.JAVA_LONG, workSize);
        invokeClMethod(
                MethodBinding.ENQUEUE_ND_RANGE_KERNEL_HANDLE,
                commandQueueMemSeg,
                kernelMemSeg,
                1,
                MemorySegment.NULL,
                workSizeMemSeg,
                MemorySegment.NULL,
                0,
                MemorySegment.NULL,
                MemorySegment.NULL);
    }

    public void finish(MemorySegment commandQueueMemSeg) {
        invokeClMethod(MethodBinding.FINISH_HANDLE, commandQueueMemSeg);
    }

    public float[] enqueueReadBuffer(MemorySegment commandQueueMemSeg, MemorySegment clBufferMemSeg, long length) {
        long byteSize = ValueLayout.JAVA_FLOAT.byteSize() * length;
        var resultMemSeg = arena.allocate(byteSize);
        invokeClMethod(
                MethodBinding.ENQUEUE_READ_BUFFER_HANDLE,
                commandQueueMemSeg,
                clBufferMemSeg,
                ParamValue.CL_TRUE,
                0L,
                byteSize,
                resultMemSeg,
                0,
                MemorySegment.NULL,
                MemorySegment.NULL);
        return resultMemSeg.toArray(ValueLayout.JAVA_FLOAT);
    }

    public void releaseClMemoryObjects(MemorySegment... clMemoryObjects) {
        for (var clMemoryObject : clMemoryObjects) {
            invokeClMethod(MethodBinding.RELEASE_MEM_OBJECT_HANDLE, clMemoryObject);
        }
    }

    public void releaseKernel(MemorySegment kernelMemSeg) {
        invokeClMethod(MethodBinding.RELEASE_KERNEL_HANDLE, kernelMemSeg);
    }

    public void releaseProgram(MemorySegment programMemSeg) {
        invokeClMethod(MethodBinding.RELEASE_PROGRAM_HANDLE, programMemSeg);
    }

    public void releaseCommandQueue(MemorySegment commandQueueMemSeg) {
        invokeClMethod(MethodBinding.RELEASE_COMMAND_QUEUE_HANDLE, commandQueueMemSeg);
    }

    public int getCommandQueueReferenceCount(MemorySegment commandQueueMemSeg) {
        var refCountMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        invokeClMethod(
                MethodBinding.GET_COMMAND_QUEUE_INFO_HANDLE,
                commandQueueMemSeg,
                ParamValue.CL_QUEUE_REFERENCE_COUNT,
                refCountMemSeg.byteSize(),
                refCountMemSeg,
                MemorySegment.NULL
        );
        return refCountMemSeg.get(ValueLayout.JAVA_INT, 0);
    }

    private MemorySegment invokeMemSegClMethod(
            MemorySegment errorCodeMemSeg, MethodHandle methodHandle, Object... arguments) {
        MemorySegment returnValue;
        try {
            returnValue = (MemorySegment) methodHandle.invokeWithArguments(arguments);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to call " + methodHandle + " with parameters: "
                    + Arrays.toString(arguments), e);
        }
        int errorCode = errorCodeMemSeg.get(ValueLayout.JAVA_INT, 0);
        if (!ReturnValue.CL_SUCCESS.matches(errorCode)) {
            throw new IllegalStateException(
                    "Error " + ReturnValue.convertToString(errorCode) + " while calling " + methodHandle);
        }
        return returnValue;
    }

    private void invokeClMethod(MethodHandle methodHandle, Object... arguments) {
        int returnValue;
        try {
            returnValue = (int) methodHandle.invokeWithArguments(arguments);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to call " + methodHandle + " with parameters: "
                    + Arrays.toString(arguments), e);
        }
        if (!ReturnValue.CL_SUCCESS.matches(returnValue)) {
            throw new IllegalStateException(
                    "Error " + ReturnValue.convertToString(returnValue) + " while calling " + methodHandle);
        }
    }

}
