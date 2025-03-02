package com.dmytrobilokha.opencl.verification;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.ProfilingInfo;
import com.dmytrobilokha.opencl.operation.AddMatricesOperation;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.foreign.ValueLayout;
import java.util.Set;

import static com.dmytrobilokha.opencl.verification.PlatformHandler.device;
import static com.dmytrobilokha.opencl.verification.PlatformHandler.platform;

@Test(groups = {"performance"})
public class AddMatricesPerformanceCheck {

    private MemoryMatrixFactory matrixFactory = MemoryMatrixFactory.newInstance();

    @DataProvider(name = "addMatricesFlavorsSizesProvider")
    public Object[][] getMatrixSizesAndKernel() {
        return new Object[][]{
                {AddMatricesOperation.Flavor.FLOAT16, 10007, 9973},
                {AddMatricesOperation.Flavor.FLOAT8, 10007, 9973},
                {AddMatricesOperation.Flavor.FLOAT4, 10007, 9973},
                {AddMatricesOperation.Flavor.FLOAT2, 10007, 9973},
                {AddMatricesOperation.Flavor.FLOAT1, 10007, 9973},
                {null, 10007, 9973},
        };
    }

    @Test(dataProvider = "addMatricesFlavorsSizesProvider")
    public void check(AddMatricesOperation.Flavor flavor, int rows, int columns) {
        var verificationMatrixA = FloatMatrix.ofUniRandoms(rows, columns);
        var verificationMatrixB = FloatMatrix.ofUniRandoms(rows, columns);
        if (flavor == null) {
            long cpuStartNanos = System.nanoTime();
            var verificationResult = verificationMatrixA.add(verificationMatrixB);
            long cpuFinishNanos = System.nanoTime();
            System.out.println(verificationResult.getRowDimension() + "X" + verificationResult.getColumnDimension()
                    + " CPU calculation microseconds: " + (cpuFinishNanos - cpuStartNanos) / 1000);
            return;
        }
        var matrixA = matrixFactory.createFloatMatrix(rows, columns);
        matrixA.setData(verificationMatrixA.getData());
        var bufferA = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * rows * columns,
                DeviceMemoryAccess.READ_ONLY,
                HostMemoryAccess.WRITE_ONLY
        );
        var matrixB = matrixFactory.createFloatMatrix(rows, columns);
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
        // TODO: enable out-of-order execution, but guarantee buffer is not read until calculation is completed
        long totalStartNanos = System.nanoTime();
        device.enqueueWriteBuffer(bufferA, matrixA);
        device.enqueueWriteBuffer(bufferB, matrixB);
        var events = addMatricesOperation.enqueue(device, Set.of());
        device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
        long totalEndNanos = System.nanoTime();
        var profilingInfos = events
                .stream()
                .map(e -> platform.getEventProfilingInfo(e))
                .toList();
        long minStartTime = profilingInfos
                .stream()
                        .mapToLong(ProfilingInfo::commandStartedNanos)
                                .min()
                .getAsLong();
        long maxFinishTime = profilingInfos
                .stream()
                .mapToLong(ProfilingInfo::commandFinishedNanos)
                .max()
                .getAsLong();
        System.out.println(rows + "X" + columns + " " + flavor.name()
                + " kernel calculation microseconds: " + (maxFinishTime - minStartTime) / 1000);
        System.out.println(rows + "X" + columns + " " + flavor.name()
                + " total microseconds: " + (totalEndNanos - totalStartNanos) / 1000);
        platform.releaseBuffer(bufferA);
        platform.releaseBuffer(bufferB);
        platform.releaseBuffer(resultBuffer);
    }

}
