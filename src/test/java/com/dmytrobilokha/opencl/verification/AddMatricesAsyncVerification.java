package com.dmytrobilokha.opencl.verification;

import com.dmytrobilokha.FileUtil;
import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.foreign.ValueLayout;
import java.util.Set;

@Test(groups = {"verification", "broken"})
public class AddMatricesAsyncVerification {

    private Platform platform;
    private Device device;
    private MemoryMatrixFactory matrixFactory;

    @BeforeMethod
    void init() {
        platform = Platform.initDefault(FileUtil.readStringResource("main.cl"));
        device = platform.getDevices().getFirst();
        matrixFactory = MemoryMatrixFactory.newInstance();
    }

    @AfterMethod
    void shutdown() {
        platform.close();
    }

    @DataProvider(name = "singleMatrixSizesProvider")
    public Object[][] getMatrixSizes() {
        return new Object[][]{
                {100, 100},
                {1000, 1000},
                {10000, 10000},
        };
    }

    @Test(dataProvider = "singleMatrixSizesProvider")
    public void addsTwoMatricesAsyncWrite(int rows, int columns) {
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
        var addMatricesKernel = platform.createKernel("addMatrices");
        platform.setKernelArgument(addMatricesKernel, 0, bufferA);
        platform.setKernelArgument(addMatricesKernel, 1, bufferB);
        platform.setKernelArgument(addMatricesKernel, 2, resultBuffer);
        platform.setKernelArgument(addMatricesKernel, 3, rows * columns);
        long nsPoint1 = System.nanoTime();
        var writeBufferAEvent = device.enqueueWriteBufferAsync(bufferA, matrixA);
        long nsPoint2 = System.nanoTime();
        var writeBufferBEvent = device.enqueueWriteBufferAsync(bufferB, matrixB);
        long nsPoint3 = System.nanoTime();
        var addMatricesEvent = device.enqueueNdRangeKernel(addMatricesKernel, 384, Set.of(writeBufferAEvent, writeBufferBEvent));
        long nsPoint4 = System.nanoTime();
        var readResultBufferEvent = device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
        long nsPoint5 = System.nanoTime();
        var verificationResult = verificationMatrixA.add(verificationMatrixB);
        long nsPoint6 = System.nanoTime();
        System.out.println(rows + "X" + columns + " async buffers write:");
        System.out.println("OpenCL calculation: " + (nsPoint5 - nsPoint1) / 1_000_000 + "ms");
        System.out.println("  writing buffer A: " + (nsPoint2 - nsPoint1) / 1_000_000 + "ms");
        TestUtil.printEventProfiling(platform.getEventProfilingInfo(writeBufferAEvent), "Writing buffer A");
        System.out.println("  writing buffer B: " + (nsPoint3 - nsPoint2) / 1_000_000 + "ms");
        TestUtil.printEventProfiling(platform.getEventProfilingInfo(writeBufferBEvent), "Writing buffer B");
        System.out.println("  execution: " + (nsPoint4 - nsPoint3) / 1_000_000 + "ms");
        TestUtil.printEventProfiling(platform.getEventProfilingInfo(addMatricesEvent), "addMatrices");
        System.out.println("  reading result buffer: " + (nsPoint5 - nsPoint4) / 1_000_000 + "ms");
        TestUtil.printEventProfiling(platform.getEventProfilingInfo(readResultBufferEvent), "Reading result buffer");
        System.out.println("CPU calculation: " + (nsPoint6 - nsPoint5) / 1_000_000 + "ms");
        TestUtil.assertMatricesEqual(resultMatrix, verificationResult);
    }

}
