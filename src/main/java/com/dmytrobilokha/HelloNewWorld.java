package com.dmytrobilokha;

import com.dmytrobilokha.opencl.ClPlatform;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

public class HelloNewWorld {

    public static void main(String[] args) {
        var kernelCode =
                "   __kernel void simple_add(__global const float* A, __global const float* B, __global float* C) { "
                        + "       C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];                      "
                        + "   }";
        try (var platform = ClPlatform.initDefault(kernelCode)) {
            var devices = platform.getDevices();
            System.out.println("Number of devices: " + devices.size());
            System.out.println("First device name: " + devices.getFirst().getName());
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
            var device = devices.getFirst();
            device.enqueueWriteBuffer(inputABuffer, inputA);
            device.enqueueWriteBuffer(inputBBuffer, inputB);
            device.enqueueNdRangeKernel(kernel, inputA.length);
            float[] result = device.enqueueReadBuffer(outputBuffer);
            System.out.println("Result is: " + Arrays.toString(result));
        }
    }

}
