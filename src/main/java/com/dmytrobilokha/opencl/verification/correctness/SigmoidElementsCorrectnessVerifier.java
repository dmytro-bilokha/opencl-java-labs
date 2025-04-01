package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.operation.SigmoidElementsOperation;
import com.dmytrobilokha.opencl.verification.FloatMatrix;
import com.dmytrobilokha.opencl.verification.MatrixSize;

import java.io.PrintWriter;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Set;

public class SigmoidElementsCorrectnessVerifier implements CorrectnessVerifier {

    private static final float ERROR_LIMIT = 2.5E-5f; // max error in %
    private static final List<MatrixSize> SIZES_TO_CHECK = List.of(
            new MatrixSize(1, 1),
            new MatrixSize(3, 5),
            new MatrixSize(1, 10),
            new MatrixSize(83, 97),
            new MatrixSize(5, 7),
            new MatrixSize(2, 1),
            new MatrixSize(1, 2),
            new MatrixSize(2, 2),
            new MatrixSize(3, 3)
    );

    public static float calculateSigmoid(float z) {
        return 1.0f / (1.0f + (float) Math.exp(-z));
    }

    @Override
    public boolean verify(Platform platform, Device device, PrintWriter reportWriter) {
        boolean allCorrect = true;
        for (var flavor : SigmoidElementsOperation.Flavor.values()) {
            for (var matrixSize : SIZES_TO_CHECK) {
                allCorrect &= executeCheck(platform, device, reportWriter, flavor, matrixSize);
            }
        }
        return allCorrect;
    }

    private boolean executeCheck(
            Platform platform,
            Device device,
            PrintWriter reportWriter,
            SigmoidElementsOperation.Flavor flavor,
            MatrixSize matrixSize
    ) {
        reportWriter.print(flavor + " " + matrixSize + ": ");
        var matrixFactory = MemoryMatrixFactory.newInstance();
        var inputMatrix = matrixFactory.createFloatMatrix(matrixSize.rows(), matrixSize.columns());
        var verificationInputMatrix = FloatMatrix.ofUniRandoms(matrixSize.rows(), matrixSize.columns());
        inputMatrix.setData(verificationInputMatrix.getData());
        var inputBuffer = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * matrixSize.numberOfElements(),
                DeviceMemoryAccess.READ_ONLY,
                HostMemoryAccess.WRITE_ONLY
        );
        var outputMatrix =  matrixFactory.createFloatMatrix(matrixSize.rows(), matrixSize.columns());
        var outputBuffer = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * matrixSize.numberOfElements(),
                DeviceMemoryAccess.WRITE_ONLY,
                HostMemoryAccess.READ_ONLY
        );
        var sigmoidElementsOperation = SigmoidElementsOperation.withFlavor(flavor, platform);
        sigmoidElementsOperation.setArguments(inputBuffer, outputBuffer, matrixSize.rows(), matrixSize.columns());
        device.enqueueWriteBuffer(inputBuffer, inputMatrix);
        sigmoidElementsOperation.enqueue(device, Set.of());
        device.enqueueReadBufferToFloatMatrix(outputBuffer, outputMatrix);
        platform.releaseBuffer(inputBuffer);
        platform.releaseBuffer(outputBuffer);
        var verificationResult = verificationInputMatrix.apply(SigmoidElementsCorrectnessVerifier::calculateSigmoid);
        float maxErrorPercent = CorrectnessVerificationUtil.calculateMaxErrorPercent(outputMatrix, verificationResult);
        reportWriter.println(maxErrorPercent + "%");
        return maxErrorPercent < ERROR_LIMIT;
    }

}
