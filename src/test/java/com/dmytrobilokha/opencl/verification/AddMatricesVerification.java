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

@Test(groups = "verification")
public class AddMatricesVerification {

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
    public void addsTwoMatrices(int rows, int columns) {
        System.out.println("Verification test");
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
        device.enqueueWriteBuffer(bufferA, matrixA);
        device.enqueueWriteBuffer(bufferB, matrixB);
        device.enqueueNdRangeKernel(addMatricesKernel, 512);
        device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
        var verificationResult = verificationMatrixA.add(verificationMatrixB);
        TestUtil.assertMatricesEqual(resultMatrix, verificationResult);
    }

}
