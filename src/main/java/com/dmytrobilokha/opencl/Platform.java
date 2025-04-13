package com.dmytrobilokha.opencl;

import com.dmytrobilokha.opencl.binding.ParamValue;
import com.dmytrobilokha.opencl.binding.MethodBinding;
import com.dmytrobilokha.opencl.binding.ReturnValue;
import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

public class Platform implements AutoCloseable {

    private static final long TMP_BUFFER_BYTE_SIZE = 250;
    private static final long TMP_BUFFER_BYTE_ALIGN = 256;

    private final Arena arena;
    private final MemorySegment tmpBufferMemSeg;
    private final MemorySegment platformIdMemSeg;
    private final String name;
    private final String version;
    private final MemorySegment deviceIdsMemSeg;
    private final MemorySegment contextMemSeg;
    private final List<Device> devices;
    private final MemorySegment programMemSeg;
    private final List<Kernel> kernels;
    private final List<PlatformBuffer> buffers;
    private final MemorySegment errorCodeMemSeg;

    public static Platform initDefault(String programSource) {
        var arena = Arena.ofConfined();
        var tmpBufferMemSeg = arena.allocate(TMP_BUFFER_BYTE_SIZE, TMP_BUFFER_BYTE_ALIGN);
        int numberOfPlatforms = queryNumberOfPlatforms(tmpBufferMemSeg);
        if (numberOfPlatforms < 1) {
            throw new OpenClRuntimeException("No OpenCL platforms found, unable to init");
        }
        var platformIdsMemSeg = queryPlatformIdsMemSeg(arena, numberOfPlatforms);
        var defaultPlatformIdMemSeg = platformIdsMemSeg.getAtIndex(ValueLayout.ADDRESS, 0);
        int numberOfPlatformDevices = queryNumberOfPlatformDevices(tmpBufferMemSeg, defaultPlatformIdMemSeg);
        if (numberOfPlatformDevices < 1) {
            throw new OpenClRuntimeException("Default platform has no devices");
        }
        var defaultPlatformName = queryPlatformName(tmpBufferMemSeg, defaultPlatformIdMemSeg);
        return new Platform(
                arena,
                tmpBufferMemSeg,
                defaultPlatformIdMemSeg,
                defaultPlatformName,
                numberOfPlatformDevices,
                programSource);
    }

    private Platform(
            Arena arena,
            MemorySegment tmpBufferMemSeg,
            MemorySegment platformIdMemSeg,
            String name,
            int numberOfPlatformDevices,
            String programSource) {
        this.arena = arena;
        this.tmpBufferMemSeg = tmpBufferMemSeg;
        this.platformIdMemSeg = platformIdMemSeg;
        this.name = name;
        this.errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        this.version = queryVersion();
        this.deviceIdsMemSeg = queryPlatformDeviceIds(numberOfPlatformDevices);
        this.contextMemSeg = createContext(numberOfPlatformDevices);
        var devicesList = new ArrayList<Device>();
        var deviceReferenceSource = new DeviceReferenceSource();
        for (int i = 0; i < numberOfPlatformDevices; i++) {
            var deviceIdMemSeg = deviceIdsMemSeg.getAtIndex(ValueLayout.ADDRESS, i);
            devicesList.add(new Device(arena, deviceReferenceSource, contextMemSeg, deviceIdMemSeg));
        }
        this.devices = List.copyOf(devicesList);
        this.programMemSeg = createProgram(programSource);
        buildProgram();
        this.kernels = new ArrayList<>();
        this.buffers = new ArrayList<>();
    }

    public Kernel createKernel(String functionName) {
        tmpBufferMemSeg.setString(0L, functionName);
        var kernelMemSeg = MethodBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_KERNEL_HANDLE,
                programMemSeg,
                tmpBufferMemSeg,
                errorCodeMemSeg
        );
        var kernel = new Kernel(functionName, kernelMemSeg);
        kernels.add(kernel);
        return kernel;
    }

    public void setKernelArgument(Kernel kernel, int argumentIndex, PlatformBuffer argument) {
        tmpBufferMemSeg.set(ValueLayout.ADDRESS, 0, argument.getBufferMemSeg());
        MethodBinding.invokeClMethod(
                MethodBinding.SET_KERNEL_ARG_HANDLE,
                kernel.getKernelMemSeg(),
                argumentIndex,
                ValueLayout.ADDRESS.byteSize(),
                tmpBufferMemSeg);
    }

    public void setKernelArgument(Kernel kernel, int argumentIndex, long argument) {
        tmpBufferMemSeg.set(ValueLayout.JAVA_LONG, 0, argument);
        MethodBinding.invokeClMethod(
                MethodBinding.SET_KERNEL_ARG_HANDLE,
                kernel.getKernelMemSeg(),
                argumentIndex,
                ValueLayout.JAVA_LONG.byteSize(),
                tmpBufferMemSeg);
    }

    public void setKernelArgument(Kernel kernel, int argumentIndex, int argument) {
        tmpBufferMemSeg.set(ValueLayout.JAVA_INT, 0, argument);
        MethodBinding.invokeClMethod(
                MethodBinding.SET_KERNEL_ARG_HANDLE,
                kernel.getKernelMemSeg(),
                argumentIndex,
                ValueLayout.JAVA_INT.byteSize(),
                tmpBufferMemSeg);
    }

    public PlatformBuffer createBuffer(long byteSize, DeviceMemoryAccess deviceAccess, HostMemoryAccess hostAccess) {
        var bufferMemSeg = MethodBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_BUFFER_HANDLE,
                contextMemSeg,
                deviceAccess.getParamValue() | hostAccess.getParamValue(),
                byteSize,
                MemorySegment.NULL,
                errorCodeMemSeg);
        var buffer = new PlatformBuffer(byteSize, bufferMemSeg, deviceAccess, hostAccess);
        buffers.add(buffer);
        return buffer;
    }

    public void releaseBuffer(PlatformBuffer buffer) {
        for (var bufferIterator = buffers.iterator(); bufferIterator.hasNext();) {
            if (buffer == bufferIterator.next()) {
                bufferIterator.remove();
                MethodBinding.invokeClMethod(MethodBinding.RELEASE_MEM_OBJECT_HANDLE, buffer.getBufferMemSeg());
                return;
            }
        }
        throw new OpenClRuntimeException("Unable to release provided buffer, it doesn't belong to the platform");
    }

    public ProfilingInfo getEventProfilingInfo(Event event) {
        long queued = getEventProfilingInfoItem(event, ParamValue.CL_PROFILING_COMMAND_QUEUED);
        long submitted = getEventProfilingInfoItem(event, ParamValue.CL_PROFILING_COMMAND_SUBMIT);
        long started = getEventProfilingInfoItem(event, ParamValue.CL_PROFILING_COMMAND_START);
        long finished = getEventProfilingInfoItem(event, ParamValue.CL_PROFILING_COMMAND_END);
        long completed = getEventProfilingInfoItem(event, ParamValue.CL_PROFILING_COMMAND_COMPLETE);
        return new ProfilingInfo(queued, submitted, started, finished, completed);
    }

    private long getEventProfilingInfoItem(Event event, int paramValue) {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_EVENT_PROFILING_INFO_HANDLE,
                event.getEventMemSeg(),
                paramValue,
                ValueLayout.JAVA_LONG.byteSize(),
                tmpBufferMemSeg,
                MemorySegment.NULL
        );
        return tmpBufferMemSeg.get(ValueLayout.JAVA_LONG, 0);
    }

    private MemorySegment queryPlatformDeviceIds(int numberOfDevices) {
        var deviceIdsMemSeg = arena.allocate(ValueLayout.ADDRESS, numberOfDevices);
        MethodBinding.invokeClMethod(
                MethodBinding.GET_DEVICE_IDS_HANDLE,
                platformIdMemSeg,
                ParamValue.CL_DEVICE_TYPE_ALL,
                numberOfDevices,
                deviceIdsMemSeg,
                MemorySegment.NULL);
        return deviceIdsMemSeg;
    }

    private MemorySegment createContext(int numberOfDevices) {
        return MethodBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_CONTEXT_HANDLE,
                MemorySegment.NULL,
                numberOfDevices,
                deviceIdsMemSeg,
                MemorySegment.NULL,
                MemorySegment.NULL,
                errorCodeMemSeg);
    }

    private long queryContextReferenceCount() {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_CONTEXT_INFO_HANDLE,
                contextMemSeg,
                ParamValue.CL_CONTEXT_REFERENCE_COUNT,
                ValueLayout.JAVA_INT.byteSize(),
                tmpBufferMemSeg,
                MemorySegment.NULL
        );
        return tmpBufferMemSeg.get(ValueLayout.JAVA_INT, 0);
    }

    private MemorySegment createProgram(String sourceCode) {
        var sourceCodeMemSeg = arena.allocateFrom(sourceCode);
        tmpBufferMemSeg.set(ValueLayout.ADDRESS, 0, sourceCodeMemSeg);
        return MethodBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_PROGRAM_WITH_SOURCE_HANDLE,
                contextMemSeg,
                devices.size(),
                tmpBufferMemSeg,
                MemorySegment.NULL,
                errorCodeMemSeg);
    }

    private void buildProgram() {
        try {
            MethodBinding.invokeClMethod(
                    MethodBinding.BUILD_PROGRAM_HANDLE,
                    programMemSeg,
                    devices.size(),
                    deviceIdsMemSeg,
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    MemorySegment.NULL
            );
        } catch (OpenClRuntimeException e) {
            if (e.getClErrorCode() == ReturnValue.CL_BUILD_PROGRAM_FAILURE) {
                throw new OpenClRuntimeException(
                        "Failed to build the program" + System.lineSeparator()
                        + devices.stream()
                                .map(d -> d.getName() + " build log:" + System.lineSeparator() + fetchProgramBuildLog(d)),
                        ReturnValue.CL_BUILD_PROGRAM_FAILURE,
                        e
                );
            }
            throw e;
        }
    }

    private String fetchProgramBuildLog(Device device) {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_PROGRAM_BUILD_INFO_HANDLE,
                programMemSeg,
                device.getDeviceIdMemSeg(),
                ParamValue.CL_PROGRAM_BUILD_LOG,
                0,
                MemorySegment.NULL,
                tmpBufferMemSeg
        );
        long logSize = tmpBufferMemSeg.get(ValueLayout.JAVA_LONG, 0);
        var logMemSeg = arena.allocate(logSize);
        MethodBinding.invokeClMethod(
                MethodBinding.GET_PROGRAM_BUILD_INFO_HANDLE,
                programMemSeg,
                device.getDeviceIdMemSeg(),
                ParamValue.CL_PROGRAM_BUILD_LOG,
                logSize,
                logMemSeg,
                MemorySegment.NULL
        );
        return logMemSeg.getString(0);
    }

    private String queryVersion() {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_PLATFORM_INFO_HANDLE,
                platformIdMemSeg,
                ParamValue.CL_PLATFORM_VERSION,
                TMP_BUFFER_BYTE_SIZE,
                tmpBufferMemSeg,
                MemorySegment.NULL
        );
        return tmpBufferMemSeg.getString(0);
    }

    private void releaseContext() {
        MethodBinding.invokeClMethod(MethodBinding.RELEASE_CONTEXT_HANDLE, contextMemSeg);
    }

    private static int queryNumberOfPlatforms(MemorySegment tmpBufferMemSeg) {
        MethodBinding.invokeClMethod(MethodBinding.GET_PLATFORM_IDS_HANDLE, 0, MemorySegment.NULL, tmpBufferMemSeg);
        return tmpBufferMemSeg.get(ValueLayout.JAVA_INT, 0);
    }

    private static MemorySegment queryPlatformIdsMemSeg(Arena arena, int numberOfPlatforms) {
        var platformIdsMemSeg = arena.allocate(ValueLayout.ADDRESS, numberOfPlatforms);
        MethodBinding.invokeClMethod(
                MethodBinding.GET_PLATFORM_IDS_HANDLE,
                numberOfPlatforms,
                platformIdsMemSeg,
                MemorySegment.NULL);
        return platformIdsMemSeg;
    }

    private static int queryNumberOfPlatformDevices(MemorySegment tmpBufferMemSeg, MemorySegment platformIdMemSeg) {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_DEVICE_IDS_HANDLE,
                platformIdMemSeg,
                ParamValue.CL_DEVICE_TYPE_ALL,
                0,
                MemorySegment.NULL,
                tmpBufferMemSeg);
        return tmpBufferMemSeg.get(ValueLayout.JAVA_INT, 0);
    }

    private static String queryPlatformName(MemorySegment tmpBufferMemSeg, MemorySegment platformIdMemSeg) {
        MethodBinding.invokeClMethod(
                MethodBinding.GET_PLATFORM_INFO_HANDLE,
                platformIdMemSeg,
                ParamValue.CL_PLATFORM_NAME,
                TMP_BUFFER_BYTE_SIZE,
                tmpBufferMemSeg,
                MemorySegment.NULL
        );
        return tmpBufferMemSeg.getString(0);
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
    }

    public List<Device> getDevices() {
        return devices;
    }

    @Override
    public void close() {
        releaseClBuffers();
        kernels.forEach(this::releaseKernel);
        releaseProgram();
        devices.forEach(Device::releaseResources);
        releaseContext();
        arena.close();
    }

    private void releaseKernel(Kernel kernel) {
        MethodBinding.invokeClMethod(MethodBinding.RELEASE_KERNEL_HANDLE, kernel.getKernelMemSeg());
    }

    private void releaseProgram() {
        MethodBinding.invokeClMethod(MethodBinding.RELEASE_PROGRAM_HANDLE, programMemSeg);
    }

    private void releaseClBuffers() {
        for (var bufferIterator = buffers.iterator(); bufferIterator.hasNext();) {
            var buffer = bufferIterator.next();
            bufferIterator.remove();
            MethodBinding.invokeClMethod(MethodBinding.RELEASE_MEM_OBJECT_HANDLE, buffer.getBufferMemSeg());
        }
    }

}
