package com.dmytrobilokha.opencl.verification;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.operation.AddMatricesOperation;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.foreign.ValueLayout;
import java.util.Set;

import static com.dmytrobilokha.opencl.verification.PlatformHandler.device;
import static com.dmytrobilokha.opencl.verification.PlatformHandler.platform;

@Test(groups = {"correctness"})
public class AddMatricesCorrectnessTest {

    private MemoryMatrixFactory matrixFactory = MemoryMatrixFactory.newInstance();

    @DataProvider(name = "addMatricesFlavorsSizesProvider")
    public Object[][] getMatrixSizesAndKernel() {
        return new Object[][]{
                {AddMatricesOperation.Flavor.FLOAT16, 10, 10},
                {AddMatricesOperation.Flavor.FLOAT8, 10, 10},
                {AddMatricesOperation.Flavor.FLOAT4, 10, 10},
                {AddMatricesOperation.Flavor.FLOAT2, 10, 10},
                {AddMatricesOperation.Flavor.FLOAT1, 10, 10},
                {AddMatricesOperation.Flavor.FLOAT16, 5, 7},
                {AddMatricesOperation.Flavor.FLOAT8, 5, 7},
                {AddMatricesOperation.Flavor.FLOAT4, 5, 7},
                {AddMatricesOperation.Flavor.FLOAT2, 5, 7},
                {AddMatricesOperation.Flavor.FLOAT1, 5, 7},
                {AddMatricesOperation.Flavor.FLOAT16, 83, 97},
                {AddMatricesOperation.Flavor.FLOAT8, 83, 97},
                {AddMatricesOperation.Flavor.FLOAT4, 83, 97},
                {AddMatricesOperation.Flavor.FLOAT2, 83, 97},
                {AddMatricesOperation.Flavor.FLOAT1, 83, 97},
                {AddMatricesOperation.Flavor.FLOAT16, 1, 10},
                {AddMatricesOperation.Flavor.FLOAT8, 1, 1},
                {AddMatricesOperation.Flavor.FLOAT4, 1, 1},
                {AddMatricesOperation.Flavor.FLOAT2, 1, 1},
                {AddMatricesOperation.Flavor.FLOAT1, 1, 1},
        };
    }

    @Test(dataProvider = "addMatricesFlavorsSizesProvider")
    public void check(AddMatricesOperation.Flavor flavor, int rows, int columns) {
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
        var addMatricesOperation = AddMatricesOperation.withFlavor(flavor, platform);
        addMatricesOperation.setArguments(bufferA, bufferB, resultBuffer, rows, columns);
        device.enqueueWriteBuffer(bufferA, matrixA);
        device.enqueueWriteBuffer(bufferB, matrixB);
        addMatricesOperation.enqueue(device, Set.of());
        device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
        platform.releaseBuffer(bufferA);
        platform.releaseBuffer(bufferB);
        platform.releaseBuffer(resultBuffer);
        var verificationResult = verificationMatrixA.add(verificationMatrixB);
        TestUtil.assertMatricesEqual(resultMatrix, verificationResult);
    }

}
