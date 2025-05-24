package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;
import com.dmytrobilokha.opencl.operation.PaddingOperation;
import com.dmytrobilokha.opencl.verification.FloatMatrix;
import com.dmytrobilokha.opencl.verification.MatrixSize;

import java.io.PrintWriter;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Set;

public class PaddingCorrectnessVerifier implements CorrectnessVerifier {

    private static final List<MatrixSize> SIZES_TO_CHECK = List.of(
            new MatrixSize(1, 1),
            new MatrixSize(3, 5),
            new MatrixSize(1, 10),
            new MatrixSize(83, 97),
            new MatrixSize(5, 7),
            new MatrixSize(2, 1),
            new MatrixSize(1, 2),
            new MatrixSize(2, 2),
            new MatrixSize(3, 3),
            new MatrixSize(8, 8),
            new MatrixSize(16, 16),
            new MatrixSize(16, 7),
            new MatrixSize(32, 32),
            new MatrixSize(19, 32),
            new MatrixSize(64, 64),
            new MatrixSize(64, 99)
    );

    @Override
    public boolean verify(Platform platform, Device device, PrintWriter reportWriter) {
        boolean allCorrect = true;
        for (var flavor : PaddingOperation.Flavor.values()) {
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
            PaddingOperation.Flavor flavor,
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
        int outputRows = (matrixSize.rows() + flavor.getTileSize() - 1) / flavor.getTileSize() * flavor.getTileSize();
        int outputColumns = (matrixSize.columns() + flavor.getTileSize() - 1) / flavor.getTileSize() * flavor.getTileSize();
        var outputMatrix = matrixFactory.createFloatMatrix(outputRows, outputColumns);
        var outputBuffer = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * outputMatrix.getNumberOfElements(),
                DeviceMemoryAccess.WRITE_ONLY,
                HostMemoryAccess.READ_ONLY
        );
        var operation = PaddingOperation.withFlavor(flavor, platform);
        operation.setArguments(inputBuffer, outputBuffer, matrixSize.rows(), matrixSize.columns(), outputRows, outputColumns);
        device.enqueueWriteBuffer(inputBuffer, inputMatrix);
        try {
            operation.enqueue(device, Set.of());
        } catch (OpenClRuntimeException e) {
            String message = e.getClErrorCode() == null
                    ? e.getMessage()
                    : e.getClErrorCode().name();
            reportWriter.println("ERROR " + message);
            platform.releaseBuffers(inputBuffer, outputBuffer);
            return true;
        }
        device.enqueueReadBufferToFloatMatrix(outputBuffer, outputMatrix);
        platform.releaseBuffers(inputBuffer, outputBuffer);
        String checkReport = CorrectnessVerificationUtil.checkMatrixProcessing(
                outputMatrix,
                (row, column) -> row < matrixSize.rows() && column < matrixSize.columns()
                        ? inputMatrix.getAt(row, column)
                        : 0f
                );
        if (checkReport.isEmpty()) {
            reportWriter.println("OK");
            return true;
        } else {
            reportWriter.print(checkReport);
            reportWriter.println("NOT OK!");
            return false;
        }
    }

}
