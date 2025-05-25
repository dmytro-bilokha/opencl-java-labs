package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;
import com.dmytrobilokha.opencl.operation.UnpaddingOperation;
import com.dmytrobilokha.opencl.verification.FloatMatrix;
import com.dmytrobilokha.opencl.verification.MatrixSize;

import java.io.PrintWriter;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class UnpaddingCorrectnessVerifier implements CorrectnessVerifier {

    private static final Map<MatrixSize, List<MatrixSize>> SIZES_TO_CHECK = Map.of(
            new MatrixSize(128, 128),
            List.of(
                    new MatrixSize(128, 128),
                    new MatrixSize(64, 128),
                    new MatrixSize(128, 64),
                    new MatrixSize(64, 64),
                    new MatrixSize(32, 32),
                    new MatrixSize(128, 123),
                    new MatrixSize(123, 128),
                    new MatrixSize(17, 19),
                    new MatrixSize(1, 1)),
            new MatrixSize(32, 32),
            List.of(
                    new MatrixSize(32, 32),
                    new MatrixSize(16, 16),
                    new MatrixSize(32, 1),
                    new MatrixSize(1, 32),
                    new MatrixSize(3, 3)
            ),
            new MatrixSize(119, 213),
            List.of(
                    new MatrixSize(100, 100),
                    new MatrixSize(13, 1),
                    new MatrixSize(22, 11)
            )
    );

    @Override
    public boolean verify(Platform platform, Device device, PrintWriter reportWriter) {
        boolean allCorrect = true;
        for (var flavor : UnpaddingOperation.Flavor.values()) {
            for (var testCase : SIZES_TO_CHECK.entrySet()) {
                allCorrect &= executeCheck(platform, device, reportWriter, flavor, testCase.getKey(), testCase.getValue());
            }
        }
        return allCorrect;
    }

    private boolean executeCheck(
            Platform platform,
            Device device,
            PrintWriter reportWriter,
            UnpaddingOperation.Flavor flavor,
            MatrixSize inputSize,
            List<MatrixSize> outputSizes
    ) {
        var matrixFactory = MemoryMatrixFactory.newInstance();
        var inputMatrix = matrixFactory.createFloatMatrix(inputSize.rows(), inputSize.columns());
        var verificationInputMatrix = FloatMatrix.ofUniRandoms(inputSize.rows(), inputSize.columns());
        inputMatrix.setData(verificationInputMatrix.getData());
        var inputBuffer = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * inputSize.numberOfElements(),
                DeviceMemoryAccess.READ_ONLY,
                HostMemoryAccess.WRITE_ONLY
        );
        device.enqueueWriteBuffer(inputBuffer, inputMatrix);
        var operation = UnpaddingOperation.withFlavor(flavor, platform);
        boolean allCorrect = true;
        for (var outputSize : outputSizes) {
            reportWriter.print(flavor + " " + inputSize + "->" + outputSize + ": ");
            var outputMatrix = matrixFactory.createFloatMatrix(outputSize.rows(), outputSize.columns());
            var outputBuffer = platform.createBuffer(
                    ValueLayout.JAVA_FLOAT.byteSize() * outputMatrix.getNumberOfElements(),
                    DeviceMemoryAccess.WRITE_ONLY,
                    HostMemoryAccess.READ_ONLY
            );
            operation.setArguments(inputBuffer, outputBuffer, inputSize.columns(), outputSize.rows(), outputSize.columns());
            try {
                operation.enqueue(device, Set.of());
            } catch (OpenClRuntimeException e) {
                String message = e.getClErrorCode() == null
                        ? e.getMessage()
                        : e.getClErrorCode().name();
                reportWriter.println("ERROR " + message);
                platform.releaseBuffer(outputBuffer);
                continue;
            }
            device.enqueueReadBufferToFloatMatrix(outputBuffer, outputMatrix);
            platform.releaseBuffer(outputBuffer);
            String checkReport = CorrectnessVerificationUtil.checkMatrixProcessing(
                    outputMatrix,
                    inputMatrix::getAt
            );
            if (checkReport.isEmpty()) {
                reportWriter.println("OK");
            } else {
                reportWriter.print(checkReport);
                reportWriter.println("NOT OK!");
                allCorrect = false;
            }

        }
        platform.releaseBuffer(inputBuffer);
        return allCorrect;
    }

}
