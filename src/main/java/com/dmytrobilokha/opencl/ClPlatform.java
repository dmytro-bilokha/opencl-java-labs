package com.dmytrobilokha.opencl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

public class ClPlatform implements AutoCloseable {

    private final Arena arena;
    private final MemorySegment platformIdMemSeg;
    private final MemorySegment deviceIdsMemSeg;
    private final MemorySegment contextMemSeg;
    private final List<ClDevice> devices;
    private final MemorySegment programMemSeg;
    private final List<Kernel> kernels;
    private final List<ClBuffer> clBuffers;

    /*
    TODO:
    - implement static method initAll(Arena) or init(name)
    - make platform auto-closable, to release all resources and arena
    - add bindings for clGetPlatformInfo to populate the name
    - add list of devices
    - query device for important params: local memory, preferable local size, etc.
    - by default create context, build program for all devices
     */

    public static ClPlatform initDefault(String programSource) {
        var arena = Arena.ofConfined();
        int numberOfPlatforms = queryNumberOfPlatforms(arena);
        if (numberOfPlatforms < 1) {
            throw new IllegalStateException("No OpenCL platforms found, unable to init");
        }
        var platformIdsMemSeg = queryPlatformIdsMemSeg(arena, numberOfPlatforms);
        var defaultPlatformIdMemSeg = platformIdsMemSeg.getAtIndex(ValueLayout.ADDRESS, 0);
        int numberOfPlatformDevices = queryNumberOfPlatformDevices(arena, defaultPlatformIdMemSeg);
        if (numberOfPlatformDevices < 1) {
            throw new IllegalStateException("Default platform has no devices");
        }
        return new ClPlatform(arena, defaultPlatformIdMemSeg, numberOfPlatformDevices, programSource);
    }

    private ClPlatform(Arena arena, MemorySegment platformIdMemSeg, int numberOfPlatformDevices, String programSource) {
        this.arena = arena;
        this.platformIdMemSeg = platformIdMemSeg;
        this.deviceIdsMemSeg = queryPlatformDeviceIds(numberOfPlatformDevices);
        this.contextMemSeg = createContext(numberOfPlatformDevices);
        var devicesList = new ArrayList<ClDevice>();
        for (int i = 0; i < numberOfPlatformDevices; i++) {
            var deviceIdMemSeg = deviceIdsMemSeg.getAtIndex(ValueLayout.ADDRESS, i);
            devicesList.add(new ClDevice(arena, contextMemSeg, deviceIdMemSeg));
        }
        this.devices = List.copyOf(devicesList);
        this.programMemSeg = createProgram(programSource);
        buildProgram();
        this.kernels = new ArrayList<>();
        this.clBuffers = new ArrayList<>();
    }

    public Kernel createKernel(String functionName) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        var kernelMemSeg = OpenClBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                OpenClBinding.CREATE_KERNEL_HANDLE,
                programMemSeg,
                arena.allocateFrom(functionName),
                errorCodeMemSeg
        );
        var kernel = new Kernel(functionName, kernelMemSeg);
        kernels.add(kernel);
        return kernel;
    }

    public void setKernelArgument(Kernel kernel, int argumentIndex, ClBuffer argument) {
        var argumentPointerMemSeg = arena.allocate(ValueLayout.ADDRESS);
        argumentPointerMemSeg.set(ValueLayout.ADDRESS, 0, argument.getBufferMemSeg());
        OpenClBinding.invokeClMethod(
                OpenClBinding.SET_KERNEL_ARG_HANDLE,
                kernel.getKernelMemSeg(),
                argumentIndex,
                argumentPointerMemSeg.byteSize(),
                argumentPointerMemSeg);
    }

    public ClBuffer createBuffer(long byteSize, DeviceMemoryAccess deviceAccess, HostMemoryAccess hostAccess) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        var bufferMemSeg = OpenClBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                OpenClBinding.CREATE_BUFFER_HANDLE,
                contextMemSeg,
                deviceAccess.getParamValue() | hostAccess.getParamValue(),
                byteSize,
                MemorySegment.NULL,
                errorCodeMemSeg);
        var buffer = new ClBuffer(byteSize, bufferMemSeg, deviceAccess, hostAccess);
        clBuffers.add(buffer);
        return buffer;
    }

    private MemorySegment queryPlatformDeviceIds(int numberOfDevices) {
        var deviceIdsMemSeg = arena.allocate(ValueLayout.ADDRESS, numberOfDevices);
        OpenClBinding.invokeClMethod(
                OpenClBinding.GET_DEVICE_IDS_HANDLE,
                platformIdMemSeg,
                ClParamValue.CL_DEVICE_TYPE_ALL,
                numberOfDevices,
                deviceIdsMemSeg,
                MemorySegment.NULL);
        return deviceIdsMemSeg;
    }

    private MemorySegment createContext(int numberOfDevices) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        return OpenClBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                OpenClBinding.CREATE_CONTEXT_HANDLE,
                MemorySegment.NULL,
                numberOfDevices,
                deviceIdsMemSeg,
                MemorySegment.NULL,
                MemorySegment.NULL,
                errorCodeMemSeg);
    }

    private MemorySegment createProgram(String sourceCode) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        var sourceCodeMemSeg = arena.allocateFrom(sourceCode);
        var pointerToSourceCodeMemSeg = arena.allocate(ValueLayout.ADDRESS);
        pointerToSourceCodeMemSeg.set(ValueLayout.ADDRESS, 0, sourceCodeMemSeg);
        return OpenClBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                OpenClBinding.CREATE_PROGRAM_WITH_SOURCE_HANDLE,
                contextMemSeg,
                devices.size(),
                pointerToSourceCodeMemSeg,
                MemorySegment.NULL,
                errorCodeMemSeg);
    }

    private void buildProgram() {
        OpenClBinding.invokeClMethod(
                OpenClBinding.BUILD_PROGRAM_HANDLE,
                programMemSeg,
                devices.size(),
                deviceIdsMemSeg,
                MemorySegment.NULL,
                MemorySegment.NULL,
                MemorySegment.NULL
        );
    }

    private void releaseContext() {
        OpenClBinding.invokeClMethod(OpenClBinding.RELEASE_CONTEXT_HANDLE, contextMemSeg);
    }

    private static int queryNumberOfPlatforms(Arena arena) {
        var numPlatformsMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        OpenClBinding.invokeClMethod(OpenClBinding.GET_PLATFORM_IDS_HANDLE, 0, MemorySegment.NULL, numPlatformsMemSeg);
        return numPlatformsMemSeg.get(ValueLayout.JAVA_INT, 0);
    }

    private static MemorySegment queryPlatformIdsMemSeg(Arena arena, int numberOfPlatforms) {
        var platformIdsMemSeg = arena.allocate(ValueLayout.ADDRESS, numberOfPlatforms);
        OpenClBinding.invokeClMethod(
                OpenClBinding.GET_PLATFORM_IDS_HANDLE,
                numberOfPlatforms,
                platformIdsMemSeg,
                MemorySegment.NULL);
        return platformIdsMemSeg;
    }

    private static int queryNumberOfPlatformDevices(Arena arena, MemorySegment platformIdMemSeg) {
        var numDevicesMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        OpenClBinding.invokeClMethod(OpenClBinding.GET_DEVICE_IDS_HANDLE,
                platformIdMemSeg,
                ClParamValue.CL_DEVICE_TYPE_ALL,
                0,
                MemorySegment.NULL,
                numDevicesMemSeg);
        return numDevicesMemSeg.get(ValueLayout.JAVA_INT, 0);
    }

    public List<ClDevice> getDevices() {
        return devices;
    }

    @Override
    public void close() {
        clBuffers.forEach(this::releaseClBuffer);
        kernels.forEach(this::releaseKernel);
        releaseProgram();
        devices.forEach(ClDevice::releaseResources);
        releaseContext();
        arena.close();
    }

    private void releaseKernel(Kernel kernel) {
        OpenClBinding.invokeClMethod(OpenClBinding.RELEASE_KERNEL_HANDLE, kernel.getKernelMemSeg());
    }

    private void releaseProgram() {
        OpenClBinding.invokeClMethod(OpenClBinding.RELEASE_PROGRAM_HANDLE, programMemSeg);
    }

    private void releaseClBuffer(ClBuffer buffer) {
        OpenClBinding.invokeClMethod(OpenClBinding.RELEASE_MEM_OBJECT_HANDLE, buffer.getBufferMemSeg());
    }

}
