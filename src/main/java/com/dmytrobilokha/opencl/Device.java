package com.dmytrobilokha.opencl;

import com.dmytrobilokha.memory.FloatMemoryMatrix;
import com.dmytrobilokha.opencl.binding.ParamValue;
import com.dmytrobilokha.opencl.binding.MethodBinding;
import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Device {

    private static final long TMP_BUFFER_BYTE_SIZE = 250;
    private static final long TMP_BUFFER_BYTE_ALIGN = 16;

    private final SegmentAllocator allocator;
    private final DeviceReferenceSource deviceReferenceSource;
    private final MemorySegment aBufferMemSeg;
    private final MemorySegment bBufferMemSeg;
    private final MemorySegment cBufferMemSeg;
    private final MemorySegment xBufferMemSeg;
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
    private final long preferredWorkGroupMultiple ;
    private final long maxClockFrequency;
    private final long maxMemoryAllocationSize;
    private final long preferredVectorWidthFloat;
    private final long max2dImageWidth;
    private final long max2dImageHeight;
    private final long numberOfCores;
    private final MemorySegment commandQueueMemSeg;

    public Device(
            SegmentAllocator allocator,
            DeviceReferenceSource deviceReferenceSource,
            MemorySegment contextMemSeg,
            MemorySegment deviceIdMemSeg
    ) {
        this.allocator = allocator;
        this.deviceReferenceSource = deviceReferenceSource;
        this.aBufferMemSeg = allocator.allocate(TMP_BUFFER_BYTE_SIZE, TMP_BUFFER_BYTE_ALIGN);
        this.bBufferMemSeg = allocator.allocate(TMP_BUFFER_BYTE_SIZE, TMP_BUFFER_BYTE_ALIGN);
        this.cBufferMemSeg = allocator.allocate(TMP_BUFFER_BYTE_SIZE, TMP_BUFFER_BYTE_ALIGN);
        this.xBufferMemSeg = allocator.allocate(TMP_BUFFER_BYTE_SIZE, TMP_BUFFER_BYTE_ALIGN);
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
        this.preferredWorkGroupMultiple = queryPreferredWorkGroupSizeMultiple();
        this.maxClockFrequency = queryMaxClockFrequency();
        this.maxMemoryAllocationSize = queryMaxMemoryAllocationSize();
        this.preferredVectorWidthFloat = queryPreferredVectorWidthFloat();
        this.max2dImageWidth = queryMax2dImageWidth();
        this.max2dImageHeight = queryMax2dImageHeight();
        this.numberOfCores = determineNumberOfCores();
        this.commandQueueMemSeg = createCommandQueue();
    }

    void releaseResources() {
        releaseCommandQueue();
    }

    private long determineNumberOfCores() {
        Long numberOfCores = deviceReferenceSource.getNumberOfCores(name);
        if (numberOfCores != null) {
            return numberOfCores;
        }
        var uppercasedName = name.toUpperCase();
        if (uppercasedName.startsWith("NVIDIA")) {
            // For Turing architecture should multiply by 64
            return maxComputeUnits * 128;
        }
        if (uppercasedName.startsWith("AMD")) {
            return maxComputeUnits * 64;
        }
        // Some wrong-for-everything default
        return maxComputeUnits * 96;
    }

    // TODO: remove this method, there are better alternatives below
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

    public Event enqueueWriteBuffer(PlatformBuffer buffer, FloatMemoryMatrix memoryMatrix) {
        if (buffer.getHostMemoryAccess() == HostMemoryAccess.NO_ACCESS
                || buffer.getHostMemoryAccess() == HostMemoryAccess.READ_ONLY) {
            throw new OpenClRuntimeException(
                    "Unable to write data to the buffer with no write access for host: " + buffer);
        }
        if (buffer.getByteSize() < memoryMatrix.getByteSize()) {
            throw new OpenClRuntimeException("Provided buffer is too small: " + buffer);
        }
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_WRITE_BUFFER_HANDLE,
                commandQueueMemSeg,
                buffer.getBufferMemSeg(),
                ParamValue.CL_TRUE,
                0L,
                memoryMatrix.getByteSize(),
                memoryMatrix.getMemorySegment(),
                0,
                MemorySegment.NULL,
                aBufferMemSeg
        );
        return Event.fromPointer(aBufferMemSeg);
    }

    public Event enqueueWriteBufferAsync(PlatformBuffer buffer, FloatMemoryMatrix memoryMatrix) {
        if (buffer.getHostMemoryAccess() == HostMemoryAccess.NO_ACCESS
                || buffer.getHostMemoryAccess() == HostMemoryAccess.READ_ONLY) {
            throw new OpenClRuntimeException(
                    "Unable to write data to the buffer with no write access for host: " + buffer);
        }
        if (buffer.getByteSize() < memoryMatrix.getByteSize()) {
            throw new OpenClRuntimeException("Provided buffer is too small: " + buffer);
        }
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_WRITE_BUFFER_HANDLE,
                commandQueueMemSeg,
                buffer.getBufferMemSeg(),
                ParamValue.CL_FALSE,
                0L,
                memoryMatrix.getByteSize(),
                memoryMatrix.getMemorySegment(),
                0,
                MemorySegment.NULL,
                aBufferMemSeg
        );
        return Event.fromPointer(aBufferMemSeg);
    }

    public Event enqueueReadBufferToFloatMatrix(PlatformBuffer buffer, FloatMemoryMatrix memoryMatrix) {
        if (buffer.getHostMemoryAccess() == HostMemoryAccess.NO_ACCESS
                || buffer.getHostMemoryAccess() == HostMemoryAccess.WRITE_ONLY) {
            throw new OpenClRuntimeException("Unable to read data from the buffer with no read access for host: "
                    + buffer);
        }
        if (buffer.getByteSize() > memoryMatrix.getByteSize()) {
            throw new OpenClRuntimeException("The buffer is bigger than matrix, unable to read");
        }
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_READ_BUFFER_HANDLE,
                commandQueueMemSeg,
                buffer.getBufferMemSeg(),
                ParamValue.CL_TRUE,
                0L,
                buffer.getByteSize(),
                memoryMatrix.getMemorySegment(),
                0,
                MemorySegment.NULL,
                aBufferMemSeg);
        return Event.fromPointer(aBufferMemSeg);
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

    public Event enqueueNdRangeKernel(Kernel kernel, long workSize) {
        aBufferMemSeg.set(ValueLayout.JAVA_LONG, 0, workSize);
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_ND_RANGE_KERNEL_HANDLE,
                commandQueueMemSeg,
                kernel.getKernelMemSeg(),
                1,
                MemorySegment.NULL,
                aBufferMemSeg,
                MemorySegment.NULL,
                0,
                MemorySegment.NULL,
                bBufferMemSeg);
        return Event.fromPointer(bBufferMemSeg);
    }

    public Event enqueueNdRangeKernel(Kernel kernel, long workSize, Set<Event> eventsToWaitFor) {
        aBufferMemSeg.set(ValueLayout.JAVA_LONG, 0, workSize);
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_ND_RANGE_KERNEL_HANDLE,
                commandQueueMemSeg,
                kernel.getKernelMemSeg(),
                1,
                MemorySegment.NULL,
                aBufferMemSeg,
                MemorySegment.NULL,
                eventsToWaitFor.size(),
                buildEventsArray(eventsToWaitFor),
                bBufferMemSeg);
        return Event.fromPointer(bBufferMemSeg);
    }

    public Event enqueueNdRangeKernel(
            Kernel kernel,
            long[] globalWorkSize,
            Set<Event> eventsToWaitFor
    ) {
        for (int i = 0; i < globalWorkSize.length; i++) {
            aBufferMemSeg.setAtIndex(ValueLayout.JAVA_LONG, i, globalWorkSize[i]);
        }
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_ND_RANGE_KERNEL_HANDLE,
                commandQueueMemSeg,
                kernel.getKernelMemSeg(),
                globalWorkSize.length,
                MemorySegment.NULL,
                aBufferMemSeg,
                MemorySegment.NULL,
                eventsToWaitFor.size(),
                buildEventsArray(eventsToWaitFor),
                bBufferMemSeg);
        return Event.fromPointer(bBufferMemSeg);
    }

    public Event enqueueNdRangeKernel(
            Kernel kernel,
            long[] localWorkSize,
            long[] globalWorkSize,
            Set<Event> eventsToWaitFor
    ) {
        if (localWorkSize.length != globalWorkSize.length) {
            throw new OpenClRuntimeException(
                    "Length of local and global work sizes should be the same, but got localWorkSize.length="
                            + localWorkSize.length
                            + " globalWorkSize.length="
                            + globalWorkSize.length);
        }
        for (int i = 0; i < globalWorkSize.length; i++) {
            aBufferMemSeg.setAtIndex(ValueLayout.JAVA_LONG, i, globalWorkSize[i]);
            bBufferMemSeg.setAtIndex(ValueLayout.JAVA_LONG, i, localWorkSize[i]);
        }
        MethodBinding.invokeClMethod(
                MethodBinding.ENQUEUE_ND_RANGE_KERNEL_HANDLE,
                commandQueueMemSeg,
                kernel.getKernelMemSeg(),
                globalWorkSize.length,
                MemorySegment.NULL,
                aBufferMemSeg,
                bBufferMemSeg,
                eventsToWaitFor.size(),
                buildEventsArray(eventsToWaitFor),
                cBufferMemSeg);
        return Event.fromPointer(cBufferMemSeg);
    }

    private MemorySegment buildEventsArray(Set<Event> events) {
        if (events.isEmpty()) {
            return MemorySegment.NULL;
        }
        var eventIterator = events.iterator();
        for (int i = 0; eventIterator.hasNext(); i++) {
            var event = eventIterator.next();
            xBufferMemSeg.setAtIndex(ValueLayout.ADDRESS, i, event.getEventMemSeg());
        }
        return xBufferMemSeg;
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
                aBufferMemSeg.byteSize(),
                aBufferMemSeg,
                MemorySegment.NULL
        );
        var result = new ArrayList<Long>();
        for (int i = 0; i < maxWorkItemDimensions; i++) {
            result.add(aBufferMemSeg.getAtIndex(ValueLayout.JAVA_LONG, i));
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

    private long queryMax2dImageWidth() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_IMAGE2D_MAX_WIDTH);
    }

    private long queryMax2dImageHeight() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_IMAGE2D_MAX_HEIGHT);
    }

    private long queryDeviceInfoLong(long paramValue) {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_DEVICE_INFO_HANDLE,
                deviceIdMemSeg,
                paramValue,
                aBufferMemSeg.byteSize(),
                aBufferMemSeg,
                MemorySegment.NULL
        );
        return aBufferMemSeg.get(ValueLayout.JAVA_LONG, 0);
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

    private long queryPreferredWorkGroupSizeMultiple() {
        return queryDeviceInfoLong(ParamValue.CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
    }

    private String queryDeviceInfoString(long paramValue) {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_DEVICE_INFO_HANDLE,
                deviceIdMemSeg,
                paramValue,
                aBufferMemSeg.byteSize(),
                aBufferMemSeg,
                MemorySegment.NULL
        );
        return aBufferMemSeg.getString(0);
    }

    private MemorySegment createCommandQueue() {
        long[] properties = new long[]{
                ParamValue.CL_QUEUE_PROPERTIES,
                ParamValue.CL_QUEUE_PROFILING_ENABLE,
                ParamValue.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                0L
        };
        MemorySegment.copy(properties, 0, aBufferMemSeg, ValueLayout.JAVA_LONG, 0L, properties.length);
        return MethodBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_COMMAND_QUEUE_WITH_PROPERTIES_HANDLE,
                contextMemSeg,
                deviceIdMemSeg,
                aBufferMemSeg,
                errorCodeMemSeg);
    }

    private void releaseCommandQueue() {
        MethodBinding.invokeClMethod(MethodBinding.RELEASE_COMMAND_QUEUE_HANDLE, commandQueueMemSeg);
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

    public long getPreferredWorkGroupMultiple() {
        return preferredWorkGroupMultiple;
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

    public long getMax2dImageWidth() {
        return max2dImageWidth;
    }

    public long getMax2dImageHeight() {
        return max2dImageHeight;
    }

    public long getNumberOfCores() {
        return numberOfCores;
    }

}
