package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.operation.AddMatricesOperation;
import com.dmytrobilokha.opencl.verification.FloatMatrix;
import com.dmytrobilokha.opencl.verification.MatrixSize;

import java.lang.foreign.ValueLayout;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class AddMatricesOperationPerformanceVerifier implements PerformanceVerifier {

    private static final List<MatrixSize> SIZES_TO_CHECK = List.of(
            new MatrixSize(10007, 9973),
            new MatrixSize(10000, 10000),
            new MatrixSize(107, 99),
            new MatrixSize(128, 128)
    );

    @Override
    public Set<PerformanceMeasurement> verify(Platform platform, Device device) {
        var result = new HashSet<PerformanceMeasurement>();
        for (var matrixSize : SIZES_TO_CHECK) {
            int rows = matrixSize.rows();
            int columns = matrixSize.columns();
            long numberOfElements = matrixSize.numberOfElements();
            var verificationMatrixA = FloatMatrix.ofUniRandoms(rows, columns);
            var verificationMatrixB = FloatMatrix.ofUniRandoms(rows, columns);
            long cpuNanoTime = determineCpuPerformance(verificationMatrixA, verificationMatrixB);
            long cpuFlops = numberOfElements * 1_000_000_000 / cpuNanoTime;
            result.add(new PerformanceMeasurement(matrixSize.toString(),"CPU", cpuFlops));
            MemoryMatrixFactory matrixFactory = MemoryMatrixFactory.newInstance();
            var matrixA = matrixFactory.createFloatMatrix(rows, columns);
            matrixA.setData(verificationMatrixA.getData());
            var bufferA = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * numberOfElements,
                    DeviceMemoryAccess.READ_ONLY,
                    HostMemoryAccess.WRITE_ONLY
            );
            var matrixB = matrixFactory.createFloatMatrix(rows, columns);
            matrixB.setData(verificationMatrixB.getData());
            var bufferB = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * numberOfElements,
                    DeviceMemoryAccess.READ_ONLY,
                    HostMemoryAccess.WRITE_ONLY
            );
            var resultMatrix =  matrixFactory.createFloatMatrix(rows, columns);
            var resultBuffer = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * numberOfElements,
                    DeviceMemoryAccess.WRITE_ONLY,
                    HostMemoryAccess.READ_ONLY
            );
            device.enqueueWriteBuffer(bufferA, matrixA);
            device.enqueueWriteBuffer(bufferB, matrixB);
            for (var flavor : AddMatricesOperation.Flavor.values()) {
                var addMatricesOperation = AddMatricesOperation.withFlavor(flavor, platform);
                addMatricesOperation.setArguments(bufferA, bufferB, resultBuffer, rows, columns);
                var events = addMatricesOperation.enqueue(device, Set.of());
                device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
                var profilingInfos = events
                        .stream()
                        .map(platform::getEventProfilingInfo)
                        .toList();
                result.add(PerformanceVerificationUtil.createMeasurement(
                        matrixSize.toString(), flavor.name(), numberOfElements, profilingInfos));
            }
            platform.releaseBuffer(bufferA);
            platform.releaseBuffer(bufferB);
            platform.releaseBuffer(resultBuffer);
        }
        return result;
    }

    @Override
    public String getName() {
        return "Add matrices";
    }

    /*
    This method determines how long does it take a CPU to add two matrices. Since JVM does multiple runtime
    optimizations, we calculate several times and take the last timing. Still, this is no way to be a precise data,
    but it is also supposed to be used only as a baseline for GPU calculation performance.
     */
    private long determineCpuPerformance(FloatMatrix a, FloatMatrix b) {
        var result1 = a.add(b);
        var result2 = b.add(a);
        long nanoBefore = System.nanoTime();
        var result3 = a.add(b);
        long nanoAfter = System.nanoTime();
        if (result1.getColumnDimension() > 0 && result2.getColumnDimension() > 0 && result3.getColumnDimension() > 0) {
            return nanoAfter - nanoBefore;
        } else {
            // This should never happen
            throw new IllegalStateException("CPU calculation has failed");
        }
    }

}
