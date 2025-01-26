package com.dmytrobilokha.opencl;

import com.dmytrobilokha.opencl.binding.ParamValue;
import com.dmytrobilokha.opencl.binding.MethodBinding;
import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;

public class Device {

    private static final long DEVICE_NAME_LIMIT = 250;

    private final SegmentAllocator allocator;
    private final MemorySegment contextMemSeg;
    private final MemorySegment deviceIdMemSeg;
    private final String name;
    private final MemorySegment commandQueueMemSeg;

    public Device(SegmentAllocator allocator, MemorySegment contextMemSeg, MemorySegment deviceIdMemSeg) {
        this.allocator = allocator;
        this.contextMemSeg = contextMemSeg;
        this.deviceIdMemSeg = deviceIdMemSeg;
        this.name = queryDeviceName();
        this.commandQueueMemSeg = createCommandQueue();
    }

    public String getName() {
        return name;
    }

    void releaseResources() {
        releaseCommandQueue();
    }

    public void enqueueWriteBuffer(PlatformBuffer buffer, float[] data) {
        long dataSize = data.length * ValueLayout.JAVA_FLOAT.byteSize();
        if (buffer.getHostMemoryAccess() == HostMemoryAccess.NO_ACCESS
            || buffer.getHostMemoryAccess() == HostMemoryAccess.READ_ONLY) {
            throw new OpenClRuntimeException(
                    "Unable to write data to the buffer with no write access for host: " + buffer);
        }
        if (buffer.getByteSize() < dataSize) {
            throw new OpenClRuntimeException("Provided buffer is too small: " + buffer);
        }
        var inputMemSeg = allocator.allocateFrom(ValueLayout.JAVA_FLOAT, data);
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_WRITE_BUFFER_HANDLE,
                commandQueueMemSeg,
                buffer.getBufferMemSeg(),
                ParamValue.CL_TRUE,
                0L,
                dataSize,
                inputMemSeg,
                0,
                MemorySegment.NULL,
                MemorySegment.NULL
        );
    }

    public float[] enqueueReadBuffer(PlatformBuffer buffer) {
        if (buffer.getHostMemoryAccess() == HostMemoryAccess.NO_ACCESS
            || buffer.getHostMemoryAccess() == HostMemoryAccess.WRITE_ONLY) {
            throw new OpenClRuntimeException("Unable to read data from the buffer with no read access for host: "
                + buffer);
        }
        var resultMemSeg = allocator.allocate(buffer.getByteSize());
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_READ_BUFFER_HANDLE,
                commandQueueMemSeg,
                buffer.getBufferMemSeg(),
                ParamValue.CL_TRUE,
                0L,
                buffer.getByteSize(),
                resultMemSeg,
                0,
                MemorySegment.NULL,
                MemorySegment.NULL);
        return resultMemSeg.toArray(ValueLayout.JAVA_FLOAT);
    }

    public void enqueueNdRangeKernel(Kernel kernel, long workSize) {
        var workSizeMemSeg = allocator.allocateFrom(ValueLayout.JAVA_LONG, workSize);
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_ND_RANGE_KERNEL_HANDLE,
                commandQueueMemSeg,
                kernel.getKernelMemSeg(),
                1,
                MemorySegment.NULL,
                workSizeMemSeg,
                MemorySegment.NULL,
                0,
                MemorySegment.NULL,
                MemorySegment.NULL);
    }

    private String queryDeviceName() {
        var deviceNameMemSeg = allocator.allocate(DEVICE_NAME_LIMIT);
        MethodBinding.invokeClMethod(
                MethodBinding.GET_DEVICE_INFO_HANDLE,
                deviceIdMemSeg,
                ParamValue.CL_DEVICE_NAME,
                DEVICE_NAME_LIMIT,
                deviceNameMemSeg,
                MemorySegment.NULL
        );
        return deviceNameMemSeg.getString(0);
    }

    private MemorySegment createCommandQueue() {
        var errorCodeMemSeg = allocator.allocate(ValueLayout.JAVA_INT);
        return MethodBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_COMMAND_QUEUE_WITH_PROPERTIES_HANDLE,
                contextMemSeg,
                deviceIdMemSeg,
                MemorySegment.NULL, //No queue properties for now
                errorCodeMemSeg);
    }

    private void releaseCommandQueue() {
        MethodBinding.invokeClMethod(MethodBinding.RELEASE_COMMAND_QUEUE_HANDLE, commandQueueMemSeg);
    }

}
