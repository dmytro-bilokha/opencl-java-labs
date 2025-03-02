package com.dmytrobilokha;

import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

public class HelloWorld {

    public static void main(String[] args) {
        try (var platform = Platform.initDefault(FileUtil.readStringResource("main.cl"))) {
            System.out.println("Default platform name: " + platform.getName());
            System.out.println("Default platform version: " + platform.getVersion());
            var devices = platform.getDevices();
            System.out.println("Number of devices: " + devices.size());
            var device = devices.getFirst();
            System.out.println("Device name: " + device.getName());
            System.out.println("Device version: " + device.getVersion());
            System.out.println("Device C version: " + device.getClangVersion());
            System.out.println("Device global memory size: " + device.getGlobalMemorySize());
            System.out.println("Device local memory size: " + device.getLocalMemorySize());
            System.out.println("Device max compute units: " + device.getMaxComputeUnits());
            System.out.println("Device max clock frequency: " + device.getMaxClockFrequency());
            System.out.println("Device max work item dimensions: " + device.getMaxWorkItemDimensions());
            System.out.println("Device max work item sizes: " + device.getMaxWorkItemSizes());
            System.out.println("Device max work group size: " + device.getMaxWorkGroupSize());
            System.out.println("Device max memory allocation size: " + device.getMaxMemoryAllocationSize());
            System.out.println("Device max 2D image width: " + device.getMax2dImageWidth());
            System.out.println("Device max 2D image height: " + device.getMax2dImageHeight());
            System.out.println("Device preferred float vector width: " + device.getPreferredVectorWidthFloat());

            float[] inputA = new float[]{1f, 3.5f, 4.0f, -11f, 42f, 99f, 0.01f, 1f, 0f, 10f};
            float[] inputB = new float[]{-1f, 13.5f, 1.0f, 1f, 0f, 1f, 100.01f, 1f, 0f, 10f};
            var inputABuffer = platform.createBuffer(inputA.length * ValueLayout.JAVA_FLOAT.byteSize(),
                    DeviceMemoryAccess.READ_ONLY, HostMemoryAccess.WRITE_ONLY);
            var inputBBuffer = platform.createBuffer(inputB.length * ValueLayout.JAVA_FLOAT.byteSize(),
                    DeviceMemoryAccess.READ_ONLY, HostMemoryAccess.WRITE_ONLY);
            var outputBuffer = platform.createBuffer(inputA.length * ValueLayout.JAVA_FLOAT.byteSize(),
                    DeviceMemoryAccess.WRITE_ONLY, HostMemoryAccess.READ_ONLY);
            var kernel = platform.createKernel("simple_add");
            platform.setKernelArgument(kernel, 0, inputABuffer);
            platform.setKernelArgument(kernel, 1, inputBBuffer);
            platform.setKernelArgument(kernel, 2, outputBuffer);
            device.enqueueWriteBuffer(inputABuffer, inputA);
            device.enqueueWriteBuffer(inputBBuffer, inputB);
            device.enqueueNdRangeKernel(kernel, inputA.length);
            float[] result = device.enqueueReadBuffer(outputBuffer);
            System.out.println("Result is: " + Arrays.toString(result));
        }
    }

}
