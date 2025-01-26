package com.dmytrobilokha.opencl;

import com.dmytrobilokha.opencl.binding.ParamValue;
import com.dmytrobilokha.opencl.binding.MethodBinding;
import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

public class Platform implements AutoCloseable {

    private final Arena arena;
    private final MemorySegment platformIdMemSeg;
    private final MemorySegment deviceIdsMemSeg;
    private final MemorySegment contextMemSeg;
    private final List<Device> devices;
    private final MemorySegment programMemSeg;
    private final List<Kernel> kernels;
    private final List<PlatformBuffer> buffers;

    /*
    TODO:
    - implement static method initAll(Arena) or init(name)
    - make platform auto-closable, to release all resources and arena
    - add bindings for clGetPlatformInfo to populate the name
    - add list of devices
    - query device for important params: local memory, preferable local size, etc.
    - by default create context, build program for all devices
     */

    public static Platform initDefault(String programSource) {
        var arena = Arena.ofConfined();
        int numberOfPlatforms = queryNumberOfPlatforms(arena);
        if (numberOfPlatforms < 1) {
            throw new OpenClRuntimeException("No OpenCL platforms found, unable to init");
        }
        var platformIdsMemSeg = queryPlatformIdsMemSeg(arena, numberOfPlatforms);
        var defaultPlatformIdMemSeg = platformIdsMemSeg.getAtIndex(ValueLayout.ADDRESS, 0);
        int numberOfPlatformDevices = queryNumberOfPlatformDevices(arena, defaultPlatformIdMemSeg);
        if (numberOfPlatformDevices < 1) {
            throw new OpenClRuntimeException("Default platform has no devices");
        }
        return new Platform(arena, defaultPlatformIdMemSeg, numberOfPlatformDevices, programSource);
    }

    private Platform(Arena arena, MemorySegment platformIdMemSeg, int numberOfPlatformDevices, String programSource) {
        this.arena = arena;
        this.platformIdMemSeg = platformIdMemSeg;
        this.deviceIdsMemSeg = queryPlatformDeviceIds(numberOfPlatformDevices);
        this.contextMemSeg = createContext(numberOfPlatformDevices);
        var devicesList = new ArrayList<Device>();
        for (int i = 0; i < numberOfPlatformDevices; i++) {
            var deviceIdMemSeg = deviceIdsMemSeg.getAtIndex(ValueLayout.ADDRESS, i);
            devicesList.add(new Device(arena, contextMemSeg, deviceIdMemSeg));
        }
        this.devices = List.copyOf(devicesList);
        this.programMemSeg = createProgram(programSource);
        buildProgram();
        this.kernels = new ArrayList<>();
        this.buffers = new ArrayList<>();
    }

    public Kernel createKernel(String functionName) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        var kernelMemSeg = MethodBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_KERNEL_HANDLE,
                programMemSeg,
                arena.allocateFrom(functionName),
                errorCodeMemSeg
        );
        var kernel = new Kernel(functionName, kernelMemSeg);
        kernels.add(kernel);
        return kernel;
    }

    public void setKernelArgument(Kernel kernel, int argumentIndex, PlatformBuffer argument) {
        var argumentPointerMemSeg = arena.allocate(ValueLayout.ADDRESS);
        argumentPointerMemSeg.set(ValueLayout.ADDRESS, 0, argument.getBufferMemSeg());
        MethodBinding.invokeClMethod(
                MethodBinding.SET_KERNEL_ARG_HANDLE,
                kernel.getKernelMemSeg(),
                argumentIndex,
                argumentPointerMemSeg.byteSize(),
                argumentPointerMemSeg);
    }

    public PlatformBuffer createBuffer(long byteSize, DeviceMemoryAccess deviceAccess, HostMemoryAccess hostAccess) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
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
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
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

    private MemorySegment createProgram(String sourceCode) {
        var errorCodeMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        var sourceCodeMemSeg = arena.allocateFrom(sourceCode);
        var pointerToSourceCodeMemSeg = arena.allocate(ValueLayout.ADDRESS);
        pointerToSourceCodeMemSeg.set(ValueLayout.ADDRESS, 0, sourceCodeMemSeg);
        return MethodBinding.invokeMemSegClMethod(
                errorCodeMemSeg,
                MethodBinding.CREATE_PROGRAM_WITH_SOURCE_HANDLE,
                contextMemSeg,
                devices.size(),
                pointerToSourceCodeMemSeg,
                MemorySegment.NULL,
                errorCodeMemSeg);
    }

    private void buildProgram() {
        MethodBinding.invokeClMethod(
                MethodBinding.BUILD_PROGRAM_HANDLE,
                programMemSeg,
                devices.size(),
                deviceIdsMemSeg,
                MemorySegment.NULL,
                MemorySegment.NULL,
                MemorySegment.NULL
        );
    }

    private void releaseContext() {
        MethodBinding.invokeClMethod(MethodBinding.RELEASE_CONTEXT_HANDLE, contextMemSeg);
    }

    private static int queryNumberOfPlatforms(Arena arena) {
        var numPlatformsMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        MethodBinding.invokeClMethod(MethodBinding.GET_PLATFORM_IDS_HANDLE, 0, MemorySegment.NULL, numPlatformsMemSeg);
        return numPlatformsMemSeg.get(ValueLayout.JAVA_INT, 0);
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

    private static int queryNumberOfPlatformDevices(Arena arena, MemorySegment platformIdMemSeg) {
        var numDevicesMemSeg = arena.allocate(ValueLayout.JAVA_INT);
        MethodBinding.invokeClMethod(MethodBinding.GET_DEVICE_IDS_HANDLE,
                platformIdMemSeg,
                ParamValue.CL_DEVICE_TYPE_ALL,
                0,
                MemorySegment.NULL,
                numDevicesMemSeg);
        return numDevicesMemSeg.get(ValueLayout.JAVA_INT, 0);
    }

    public List<Device> getDevices() {
        return devices;
    }

    @Override
    public void close() {
        buffers.forEach(this::releaseClBuffer);
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

    private void releaseClBuffer(PlatformBuffer buffer) {
        MethodBinding.invokeClMethod(MethodBinding.RELEASE_MEM_OBJECT_HANDLE, buffer.getBufferMemSeg());
    }

}
