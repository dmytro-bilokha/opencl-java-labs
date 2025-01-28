package com.dmytrobilokha.opencl;

import com.dmytrobilokha.opencl.binding.ParamValue;
import com.dmytrobilokha.opencl.binding.MethodBinding;
import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

public class Device {

    private static final long TMP_BUFFER_BYTE_SIZE = 250;
    private static final long TMP_BUFFER_BYTE_ALIGN = 256;

    private final SegmentAllocator allocator;
    private final MemorySegment tmpBufferMemSeg;
    private final MemorySegment contextMemSeg;
    private final MemorySegment deviceIdMemSeg;
    private final MemorySegment errorCodeMemSeg;
    private final String name;
    private final String version;
    private final String clangVersion;
    private final long globalMemorySize;
    private final long localMemorySize;
    private final long maxComputeUnits;
    private final long maxWorkItemDimensions;
    private final List<Long> maxWorkItemSizes;
    private final long maxWorkGroupSize;
    private final long maxClockFrequency;
    private final long maxMemoryAllocationSize;
    private final long preferredVectorWidthFloat;
    private final MemorySegment commandQueueMemSeg;

    public Device(SegmentAllocator allocator, MemorySegment contextMemSeg, MemorySegment deviceIdMemSeg) {
        this.allocator = allocator;
        this.tmpBufferMemSeg = allocator.allocate(TMP_BUFFER_BYTE_SIZE, TMP_BUFFER_BYTE_ALIGN);
        this.contextMemSeg = contextMemSeg;
        this.deviceIdMemSeg = deviceIdMemSeg;
        this.errorCodeMemSeg = allocator.allocate(ValueLayout.JAVA_INT);
        this.name = queryDeviceName();
        this.version = queryVersion();
        this.clangVersion = queryClangVersion();
        this.globalMemorySize = queryGlobalMemorySize();
        this.localMemorySize = queryLocalMemorySize();
        this.maxComputeUnits = queryMaxComputeUnits();
        this.maxWorkItemDimensions = queryMaxWorkItemDimensions();
        this.maxWorkItemSizes = List.copyOf(queryMaxWorkItemSizes());
        this.maxWorkGroupSize = queryMaxWorkGroupSize();
        this.maxClockFrequency = queryMaxClockFrequency();
        this.maxMemoryAllocationSize = queryMaxMemoryAllocationSize();
        this.preferredVectorWidthFloat = queryPreferredVectorWidthFloat();
        this.commandQueueMemSeg = createCommandQueue();
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

    public long getGlobalMemorySize() {
        return globalMemorySize;
    }

    public long getLocalMemorySize() {
        return localMemorySize;
    }

    public long getMaxComputeUnits() {
        return maxComputeUnits;
    }

    public long getMaxWorkItemDimensions() {
        return maxWorkItemDimensions;
    }

    public List<Long> getMaxWorkItemSizes() {
        return maxWorkItemSizes;
    }

    public long getMaxWorkGroupSize() {
        return maxWorkGroupSize;
    }

    public long getMaxClockFrequency() {
        return maxClockFrequency;
    }

    public long getMaxMemoryAllocationSize() {
        return maxMemoryAllocationSize;
    }

    public long getPreferredVectorWidthFloat() {
        return preferredVectorWidthFloat;
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
    }

    public String getClangVersion() {
        return clangVersion;
    }

    private long queryGlobalMemorySize() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_GLOBAL_MEM_SIZE);
    }

    private long queryLocalMemorySize() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_LOCAL_MEM_SIZE);
    }

    private long queryMaxComputeUnits() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    private long queryMaxWorkItemDimensions() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    }

    private List<Long> queryMaxWorkItemSizes() {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_DEVICE_INFO_HANDLE,
                deviceIdMemSeg,
                ParamValue.CL_DEVICE_MAX_WORK_ITEM_SIZES,
                tmpBufferMemSeg.byteSize(),
                tmpBufferMemSeg,
                MemorySegment.NULL
        );
        var result = new ArrayList<Long>();
        for (int i = 0; i < maxWorkItemDimensions; i++) {
            result.add(tmpBufferMemSeg.getAtIndex(ValueLayout.JAVA_LONG, i));
        }
        return result;
    }

    private long queryMaxWorkGroupSize() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    private long queryPreferredVectorWidthFloat() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
    }

    private long queryMaxClockFrequency() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    private long queryMaxMemoryAllocationSize() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    }

    private long queryDeviceInfoLong(long paramValue) {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_DEVICE_INFO_HANDLE,
                deviceIdMemSeg,
                paramValue,
                tmpBufferMemSeg.byteSize(),
                tmpBufferMemSeg,
                MemorySegment.NULL
        );
        return tmpBufferMemSeg.get(ValueLayout.JAVA_LONG, 0);
    }

    private String queryDeviceName() {
        return queryDeviceInfoString(ParamValue.CL_DEVICE_NAME);
    }

    private String queryVersion() {
        return queryDeviceInfoString(ParamValue.CL_DEVICE_VERSION);
    }

    private String queryClangVersion() {
        return queryDeviceInfoString(ParamValue.CL_DEVICE_OPENCL_C_VERSION);
    }

    private String queryDeviceInfoString(long paramValue) {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_DEVICE_INFO_HANDLE,
                deviceIdMemSeg,
                paramValue,
                tmpBufferMemSeg.byteSize(),
                tmpBufferMemSeg,
                MemorySegment.NULL
        );
        return tmpBufferMemSeg.getString(0);
    }

    private MemorySegment createCommandQueue() {
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
