package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.memory.FloatMemoryMatrix;
import com.dmytrobilokha.opencl.verification.FloatMatrix;

public final class CorrectnessVerificationUtil {

    private static final int ERRORS_LIMIT = 5;

    private CorrectnessVerificationUtil() {
        // Util, no instance
    }

    public static String checkMatricesEqual(FloatMemoryMatrix actual, FloatMatrix expected) {
        float[][] verificationData = expected.getData();
        int errorsCount = 0;
        StringBuilder errorReportBuilder = new StringBuilder();
        for (int i = 0; i < expected.getRowDimension(); i++) {
            float[] row = verificationData[i];
            for (int j = 0; j < expected.getColumnDimension(); j++) {
                float expectedValue = row[j];
                float actualValue = actual.getAt(i, j);
                if (actualValue != expectedValue) {
                    errorsCount++;
                    if (errorsCount <= ERRORS_LIMIT) {
                        errorReportBuilder
                                .append("For (")
                                .append(i)
                                .append(", ")
                                .append(j)
                                .append(") expected ")
                                .append(expected)
                                .append(", but got ")
                                .append(actualValue)
                                .append(System.lineSeparator());
                    } else {
                        return errorReportBuilder.toString();
                    }
                }
            }
        }
        return errorReportBuilder.toString();
    }

}
