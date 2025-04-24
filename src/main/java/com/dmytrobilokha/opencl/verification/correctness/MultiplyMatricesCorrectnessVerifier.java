package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.operation.MultiplyMatricesOperation;
import com.dmytrobilokha.opencl.verification.FloatMatrix;
import com.dmytrobilokha.opencl.verification.MatricesMultiplicationSize;

import java.io.PrintWriter;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Set;

public class MultiplyMatricesCorrectnessVerifier implements CorrectnessVerifier {

    private static final float ERROR_LIMIT = 1.2E-4f; // max error in %

    // TODO: add more "tricky" sized test-cases
    private static final List<MatricesMultiplicationSize> SIZES_TO_CHECK = List.of(
            /* Tiled implementation can not handle such sizes
            new MatricesMultiplicationSize(1, 1, 1),
            new MatricesMultiplicationSize(3, 3, 3),
            new MatricesMultiplicationSize(5, 5, 5),
            new MatricesMultiplicationSize(16, 16, 16),
            new MatricesMultiplicationSize(32, 32, 32),
             */
            new MatricesMultiplicationSize(64, 64, 64),
            new MatricesMultiplicationSize(128, 128, 128),
            new MatricesMultiplicationSize(256, 256, 256)
    );

    @Override
    public boolean verify(Platform platform, Device device, PrintWriter reportWriter) {
        boolean allCorrect = true;
        for (var flavor : MultiplyMatricesOperation.Flavor.values()) {
            for (var size : SIZES_TO_CHECK) {
                allCorrect &= executeCheck(platform, device, reportWriter, flavor, size);
            }
        }
        return allCorrect;
    }

    private boolean executeCheck(
            Platform platform,
            Device device,
            PrintWriter reportWriter,
            MultiplyMatricesOperation.Flavor flavor,
            MatricesMultiplicationSize size
    ) {
        reportWriter.print(flavor + " " + size + ": ");
        var matrixFactory = MemoryMatrixFactory.newInstance();
        var matrixA = matrixFactory.createFloatMatrix(size.mDimension(), size.kDimension());
        var verificationMatrixA = FloatMatrix.ofUniRandoms(size.mDimension(), size.kDimension());
        matrixA.setData(verificationMatrixA.getData());
        var bufferA = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * matrixA.getNumberOfElements(),
                DeviceMemoryAccess.READ_ONLY,
                HostMemoryAccess.WRITE_ONLY
        );
        var matrixB = matrixFactory.createFloatMatrix(size.kDimension(), size.nDimension());
        var verificationMatrixB = FloatMatrix.ofUniRandoms(size.kDimension(), size.nDimension());
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
        var operation = MultiplyMatricesOperation.withFlavor(flavor, platform);
        operation.setArguments(bufferA, bufferB, resultBuffer, size.mDimension(), size.kDimension(), size.nDimension());
        device.enqueueWriteBuffer(bufferA, matrixA);
        device.enqueueWriteBuffer(bufferB, matrixB);
        operation.enqueue(device, Set.of());
        device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
        platform.releaseBuffer(bufferA);
        platform.releaseBuffer(bufferB);
        platform.releaseBuffer(resultBuffer);
        var verificationResult = verificationMatrixA.multiply(verificationMatrixB);
        float maxErrorPercent = CorrectnessVerificationUtil.calculateMaxErrorPercent(resultMatrix, verificationResult);
        reportWriter.println(maxErrorPercent + "%");
        return maxErrorPercent < ERROR_LIMIT;
    }

}
