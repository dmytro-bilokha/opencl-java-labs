package com.dmytrobilokha;

import com.dmytrobilokha.opencl.OpenClConnector;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

public class HelloWorld {

    public static void main(String[] cliArgs) {
        try (var offHeap = Arena.ofConfined()) {
            var openClConnector = new OpenClConnector(offHeap);
            var defaultDeviceIdsMemSeg = openClConnector.getDefaultPlatformDeviceIds();
            System.out.println("Default device IDs: " + defaultDeviceIdsMemSeg);
            var defaultDeviceIdMemSeg = defaultDeviceIdsMemSeg.get(ValueLayout.ADDRESS, 0);
            System.out.println("Device name: " + openClConnector.getDeviceName(defaultDeviceIdMemSeg));
            var contextMemSeg = openClConnector.createContext(defaultDeviceIdsMemSeg);
            float[] inputA = new float[]{1f, 3.5f, 4.0f, -11f, 42f, 99f, 0.01f, 1f, 0f, 10f};
            float[] inputB = new float[]{-1f, 13.5f, 1.0f, 1f, 0f, 1f, 100.01f, 1f, 0f, 10f};
            var inputAMemSeg = openClConnector.createInputBufferOfFloats(contextMemSeg, inputA);
            var inputBMemSeg = openClConnector.createInputBufferOfFloats(contextMemSeg, inputB);
            long outputByteSize = ValueLayout.JAVA_FLOAT.byteSize() * inputA.length;
            var outputMemSeg = openClConnector.createOutputBuffer(contextMemSeg, outputByteSize);
            var commandQueueMemSeg = openClConnector.createCommandQueue(contextMemSeg, defaultDeviceIdMemSeg);
            var kernelCode =
                    "   __kernel void simple_add(__global const float* A, __global const float* B, __global float* C) { "
                            + "       C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];                      "
                            + "   }";
            var programMemSeg = openClConnector.createProgram(contextMemSeg, kernelCode);
            openClConnector.buildProgram(programMemSeg, defaultDeviceIdsMemSeg);
            var kernelMemSeg = openClConnector.createKernel(programMemSeg, "simple_add");
            openClConnector.setKernelArgument(kernelMemSeg, 0, inputAMemSeg);
            openClConnector.setKernelArgument(kernelMemSeg, 1, inputBMemSeg);
            openClConnector.setKernelArgument(kernelMemSeg, 2, outputMemSeg);
            openClConnector.enqueueNdRangeKernel(commandQueueMemSeg, kernelMemSeg, inputA.length);
            openClConnector.finish(commandQueueMemSeg);
            float[] result = openClConnector.enqueueReadBuffer(commandQueueMemSeg, outputMemSeg, inputA.length);
            System.out.println("Result is: " + Arrays.toString(result));
            openClConnector.releaseClMemoryObjects(inputAMemSeg, inputBMemSeg, outputMemSeg);
            openClConnector.releaseKernel(kernelMemSeg);
            openClConnector.releaseProgram(programMemSeg);
            openClConnector.releaseCommandQueue(commandQueueMemSeg);
            openClConnector.releaseContext(contextMemSeg);
        }
    }

}
