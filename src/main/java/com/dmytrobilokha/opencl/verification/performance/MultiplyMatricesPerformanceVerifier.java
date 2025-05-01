package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;
import com.dmytrobilokha.opencl.operation.MultiplyMatricesOperation;
import com.dmytrobilokha.opencl.verification.FloatMatrix;
import com.dmytrobilokha.opencl.verification.MatricesMultiplicationSize;

import java.lang.foreign.ValueLayout;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class MultiplyMatricesPerformanceVerifier implements PerformanceVerifier {

    private static final List<MatricesMultiplicationSize> SIZES_TO_CHECK = List.of(
            new MatricesMultiplicationSize(64, 64, 64),
            new MatricesMultiplicationSize(128, 128, 128),
            new MatricesMultiplicationSize(256, 256, 256),
            new MatricesMultiplicationSize(512, 512, 512),
            new MatricesMultiplicationSize(1024, 1024, 1024),
            new MatricesMultiplicationSize(2048, 2048, 2048)
    );

    @Override
    public Set<PerformanceMeasurement> verify(Platform platform, Device device) {
        var result = new HashSet<PerformanceMeasurement>();
        for (var size : SIZES_TO_CHECK) {
            var verificationMatrixA = FloatMatrix.ofUniRandoms(size.mDimension(), size.kDimension());
            var verificationMatrixB = FloatMatrix.ofUniRandoms(size.kDimension(), size.nDimension());
            long cpuNanoTime = determineCpuPerformance(verificationMatrixA, verificationMatrixB);
            long numberOfOperations = size.mDimension() * size.nDimension() * (2L * size.kDimension() - 1);
            long cpuFlops = PerformanceVerificationUtil.calculateFlops(numberOfOperations, cpuNanoTime);
            result.add(new PerformanceMeasurement(size.toString(),"CPU", cpuFlops));
            MemoryMatrixFactory matrixFactory = MemoryMatrixFactory.newInstance();
            var matrixA = matrixFactory.createFloatMatrix(size.mDimension(), size.kDimension());
            matrixA.setData(verificationMatrixA.getData());
            var bufferA = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * matrixA.getNumberOfElements(),
                    DeviceMemoryAccess.READ_ONLY,
                    HostMemoryAccess.WRITE_ONLY
            );
            var matrixB = matrixFactory.createFloatMatrix(size.kDimension(), size.nDimension());
            matrixB.setData(verificationMatrixB.getData());
            var bufferB = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * matrixB.getNumberOfElements(),
                    DeviceMemoryAccess.READ_ONLY,
                    HostMemoryAccess.WRITE_ONLY
            );
            var resultMatrix =  matrixFactory.createFloatMatrix(size.mDimension(), size.nDimension());
            var resultBuffer = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * resultMatrix.getNumberOfElements(),
                    DeviceMemoryAccess.WRITE_ONLY,
                    HostMemoryAccess.READ_ONLY
            );
            device.enqueueWriteBuffer(bufferA, matrixA);
            device.enqueueWriteBuffer(bufferB, matrixB);
            for (var flavor : MultiplyMatricesOperation.Flavor.values()) {
                try {
                    var operation = MultiplyMatricesOperation.withFlavor(flavor, platform);
                    operation.setArguments(bufferA, bufferB, resultBuffer, size.mDimension(), size.kDimension(), size.nDimension());
                    var events = operation.enqueue(device, Set.of());
                    device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
                    var profilingInfos = events
                            .stream()
                            .map(platform::getEventProfilingInfo)
                            .toList();
                    result.add(PerformanceVerificationUtil.createMeasurement(
                            size.toString(), flavor.name(), numberOfOperations, profilingInfos));
                } catch (OpenClRuntimeException e) {
                    result.add(PerformanceMeasurement.ofFailure(size.toString(), flavor.name(), e));
                }
            }
            platform.releaseBuffer(bufferA);
            platform.releaseBuffer(bufferB);
            platform.releaseBuffer(resultBuffer);
        }
        return result;
    }

    @Override
    public String getName() {
        return "Multiply matrices";
    }

    /*
    This method determines how long does it take a CPU to multiply two matrices. Since JVM does multiple runtime
    optimizations, we calculate several times and take the last timing. Still, this is no way to be a precise data,
    but it is also supposed to be used only as a baseline for GPU calculation performance.
     */
    private long determineCpuPerformance(FloatMatrix a, FloatMatrix b) {
        var result1 = a.multiply(b);
        var result2 = b.multiply(a);
        long nanoBefore = System.nanoTime();
        var result3 = a.multiply(b);
        long nanoAfter = System.nanoTime();
        if (result1.getColumnDimension() > 0 && result2.getColumnDimension() > 0 && result3.getColumnDimension() > 0) {
            return nanoAfter - nanoBefore;
        } else {
            // This should never happen
            throw new IllegalStateException("CPU calculation has failed");
        }
    }

}
