package com.dmytrobilokha.opencl.verification;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.foreign.ValueLayout;

import static com.dmytrobilokha.opencl.verification.PlatformHandler.device;
import static com.dmytrobilokha.opencl.verification.PlatformHandler.platform;

@Test(groups = {"verification"})
public class AddMatricesVerification {

    private MemoryMatrixFactory matrixFactory = MemoryMatrixFactory.newInstance();

    @DataProvider(name = "singleMatrixSizesKernelsProvider")
    public Object[][] getMatrixSizesAndKernel() {
        return new Object[][]{
                {"addMatrices", 100, 100},
                {"addMatrices", 1000, 1000},
                {"addMatrices", 10000, 10000},
                {"addMatrices", 10000, 10000},
                {"addMatrices", 10000, 10000},
        };
    }

    @Test(dataProvider = "singleMatrixSizesKernelsProvider")
    public void addsTwoMatrices(String kernel, int rows, int columns) {
        var matrixA = matrixFactory.createFloatMatrix(rows, columns);
        var verificationMatrixA = FloatMatrix.ofUniRandoms(rows, columns);
        matrixA.setData(verificationMatrixA.getData());
        var bufferA = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * rows * columns,
                DeviceMemoryAccess.READ_ONLY,
                HostMemoryAccess.WRITE_ONLY
        );
        var matrixB = matrixFactory.createFloatMatrix(rows, columns);
        var verificationMatrixB = FloatMatrix.ofUniRandoms(rows, columns);
        matrixB.setData(verificationMatrixB.getData());
        var bufferB = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * rows * columns,
                DeviceMemoryAccess.READ_ONLY,
                HostMemoryAccess.WRITE_ONLY
        );
        var resultMatrix =  matrixFactory.createFloatMatrix(rows, columns);
        var resultBuffer = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * rows * columns,
                DeviceMemoryAccess.WRITE_ONLY,
                HostMemoryAccess.READ_ONLY
        );
        var addMatricesKernel = platform.createKernel(kernel);
        platform.setKernelArgument(addMatricesKernel, 0, bufferA);
        platform.setKernelArgument(addMatricesKernel, 1, bufferB);
        platform.setKernelArgument(addMatricesKernel, 2, resultBuffer);
        platform.setKernelArgument(addMatricesKernel, 3, rows * columns);
        long nsPoint1 = System.nanoTime();
        var writeBufferAEvent = device.enqueueWriteBuffer(bufferA, matrixA);
        long nsPoint2 = System.nanoTime();
        var writeBufferBEvent = device.enqueueWriteBuffer(bufferB, matrixB);
        long nsPoint3 = System.nanoTime();
        var addMatricesEvent = device.enqueueNdRangeKernel(addMatricesKernel, 384);
        long nsPoint4 = System.nanoTime();
        var readResultBufferEvent = device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
        long nsPoint5 = System.nanoTime();
        var verificationResult = verificationMatrixA.add(verificationMatrixB);
        long nsPoint6 = System.nanoTime();
        System.out.println(rows + "X" + columns + ":");
        System.out.println("OpenCL calculation kernel " + kernel + ": " + (nsPoint5 - nsPoint1) / 1_000_000 + "ms");
        System.out.println("  writing buffer A: " + (nsPoint2 - nsPoint1) / 1_000_000 + "ms");
        TestUtil.printEventProfiling(platform.getEventProfilingInfo(writeBufferAEvent), "Writing buffer A");
        System.out.println("  writing buffer B: " + (nsPoint3 - nsPoint2) / 1_000_000 + "ms");
        TestUtil.printEventProfiling(platform.getEventProfilingInfo(writeBufferBEvent), "Writing buffer B");
        System.out.println("  execution: " + (nsPoint4 - nsPoint3) / 1_000_000 + "ms");
        TestUtil.printEventProfiling(platform.getEventProfilingInfo(addMatricesEvent), kernel);
        System.out.println("  reading result buffer: " + (nsPoint5 - nsPoint4) / 1_000_000 + "ms");
        TestUtil.printEventProfiling(platform.getEventProfilingInfo(readResultBufferEvent), "Reading result buffer");
        System.out.println("CPU calculation: " + (nsPoint6 - nsPoint5) / 1_000_000 + "ms");
        platform.releaseBuffer(bufferA);
        platform.releaseBuffer(bufferB);
        platform.releaseBuffer(resultBuffer);
        TestUtil.assertMatricesEqual(resultMatrix, verificationResult);
    }

}
