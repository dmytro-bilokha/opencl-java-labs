package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.memory.MemoryMatrixFactory;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.DeviceMemoryAccess;
import com.dmytrobilokha.opencl.HostMemoryAccess;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.operation.AddMatricesOperation;
import com.dmytrobilokha.opencl.verification.FloatMatrix;
import com.dmytrobilokha.opencl.verification.MatrixSize;

import java.io.PrintWriter;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Set;

public class AddMatricesOperationCorrectnessVerifier implements CorrectnessVerifier {


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

    @Override
    public boolean verify(Platform platform, Device device, PrintWriter reportWriter) {
        boolean allCorrect = true;
        for (var flavor : AddMatricesOperation.Flavor.values()) {
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
            AddMatricesOperation.Flavor flavor,
            MatrixSize matrixSize
    ) {
        reportWriter.print(flavor + " " + matrixSize.rows() + "X" + matrixSize.columns() + ": ");
        var matrixFactory = MemoryMatrixFactory.newInstance();
        var matrixA = matrixFactory.createFloatMatrix(matrixSize.rows(), matrixSize.columns());
        var verificationMatrixA = FloatMatrix.ofUniRandoms(matrixSize.rows(), matrixSize.columns());
        matrixA.setData(verificationMatrixA.getData());
        var bufferA = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * matrixSize.numberOfElements(),
                DeviceMemoryAccess.READ_ONLY,
                HostMemoryAccess.WRITE_ONLY
        );
        var matrixB = matrixFactory.createFloatMatrix(matrixSize.rows(), matrixSize.columns());
        var verificationMatrixB = FloatMatrix.ofUniRandoms(matrixSize.rows(), matrixSize.columns());
        matrixB.setData(verificationMatrixB.getData());
        var bufferB = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * matrixSize.numberOfElements(),
                DeviceMemoryAccess.READ_ONLY,
                HostMemoryAccess.WRITE_ONLY
        );
        var resultMatrix =  matrixFactory.createFloatMatrix(matrixSize.rows(), matrixSize.columns());
        var resultBuffer = platform.createBuffer(
                ValueLayout.JAVA_FLOAT.byteSize() * matrixSize.numberOfElements(),
                DeviceMemoryAccess.WRITE_ONLY,
                HostMemoryAccess.READ_ONLY
        );
        var addMatricesOperation = AddMatricesOperation.withFlavor(flavor, platform);
        addMatricesOperation.setArguments(bufferA, bufferB, resultBuffer, matrixSize.rows(), matrixSize.columns());
        device.enqueueWriteBuffer(bufferA, matrixA);
        device.enqueueWriteBuffer(bufferB, matrixB);
        addMatricesOperation.enqueue(device, Set.of());
        device.enqueueReadBufferToFloatMatrix(resultBuffer, resultMatrix);
        platform.releaseBuffer(bufferA);
        platform.releaseBuffer(bufferB);
        platform.releaseBuffer(resultBuffer);
        var verificationResult = verificationMatrixA.add(verificationMatrixB);
        String checkReport = CorrectnessVerificationUtil.checkMatricesEqual(resultMatrix, verificationResult);
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
