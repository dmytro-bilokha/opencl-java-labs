package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.operation.SigmoidElementsOperation;
import com.dmytrobilokha.opencl.verification.FloatMatrix;
import com.dmytrobilokha.opencl.verification.MatrixSize;
import com.dmytrobilokha.opencl.verification.correctness.SigmoidElementsCorrectnessVerifier;

import java.lang.foreign.ValueLayout;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class SigmoidElementsPerformanceVerifier implements PerformanceVerifier {

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
            long numberOfOperations = 4 * numberOfElements;
            var verificationInput = FloatMatrix.ofUniRandoms(rows, columns);
            long cpuNanoTime = determineCpuPerformance(verificationInput);
            long cpuFlops = numberOfOperations * 1_000_000_000 / cpuNanoTime;
            result.add(new PerformanceMeasurement(matrixSize.toString(),"CPU", cpuFlops));
            MemoryMatrixFactory matrixFactory = MemoryMatrixFactory.newInstance();
            var inputMatrix = matrixFactory.createFloatMatrix(rows, columns);
            inputMatrix.setData(verificationInput.getData());
            var inputBuffer = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * numberOfElements,
                    DeviceMemoryAccess.READ_ONLY,
                    HostMemoryAccess.WRITE_ONLY
            );
            var outputMatrix =  matrixFactory.createFloatMatrix(rows, columns);
            var outputBuffer = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * numberOfElements,
                    DeviceMemoryAccess.WRITE_ONLY,
                    HostMemoryAccess.READ_ONLY
            );
            device.enqueueWriteBuffer(inputBuffer, inputMatrix);
            for (var flavor : SigmoidElementsOperation.Flavor.values()) {
                var operation = SigmoidElementsOperation.withFlavor(flavor, platform);
                operation.setArguments(inputBuffer, outputBuffer, rows, columns);
                var events = operation.enqueue(device, Set.of());
                device.enqueueReadBufferToFloatMatrix(outputBuffer, outputMatrix);
                var profilingInfos = events
                        .stream()
                        .map(platform::getEventProfilingInfo)
                        .toList();
                result.add(PerformanceVerificationUtil.createMeasurement(
                        matrixSize.toString(), flavor.name(), numberOfOperations, profilingInfos));
            }
            platform.releaseBuffer(inputBuffer);
            platform.releaseBuffer(outputBuffer);
        }
        return result;
    }

    @Override
    public String getName() {
        return "Sigmoid function";
    }

    /*
    This method determines how long does it take a CPU to calculate sigmoid over elements of a matrix.
    Since JVM does multiple runtime optimizations, we calculate several times and take the last timing.
    Still, this is no way to be a precise data, but it is also supposed to be used only as a baseline
    for GPU calculation performance.
     */
    private long determineCpuPerformance(FloatMatrix input) {
        var result1 = input.apply(SigmoidElementsCorrectnessVerifier::calculateSigmoid);
        var result2 = input.apply(SigmoidElementsCorrectnessVerifier::calculateSigmoid);
        long nanoBefore = System.nanoTime();
        var result3 = input.apply(SigmoidElementsCorrectnessVerifier::calculateSigmoid);
        long nanoAfter = System.nanoTime();
        if (result1.getColumnDimension() > 0 && result2.getColumnDimension() > 0 && result3.getColumnDimension() > 0) {
            return nanoAfter - nanoBefore;
        } else {
            // This should never happen
            throw new IllegalStateException("CPU calculation has failed");
        }
    }

}
