package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.memory.FloatMemoryMatrix;
import com.dmytrobilokha.opencl.verification.FloatMatrix;

public final class CorrectnessVerificationUtil {

    private static final float EPSILON = 1E-7f;
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
                                .append(expectedValue)
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

    public static String checkMatrixProcessing(FloatMemoryMatrix output, FloatElementProvider expectedElementProvider) {
        int errorsCount = 0;
        StringBuilder errorReportBuilder = new StringBuilder();
        for (int i = 0; i < output.getRows(); i++) {
            for (int j = 0; j < output.getColumns(); j++) {
                float expectedValue = expectedElementProvider.provide(i, j);
                float actualValue = output.getAt(i, j);
                if (actualValue != expectedValue) {
                    errorsCount++;
                    if (errorsCount <= ERRORS_LIMIT) {
                        errorReportBuilder
                                .append("For (")
                                .append(i)
                                .append(", ")
                                .append(j)
                                .append(") expected ")
                                .append(expectedValue)
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

    public static float calculateMaxErrorPercent(FloatMemoryMatrix actual, FloatMatrix expected) {
        float[][] verificationData = expected.getData();
        float maxErrorPercent = 0;
        for (int i = 0; i < expected.getRowDimension(); i++) {
            float[] row = verificationData[i];
            for (int j = 0; j < expected.getColumnDimension(); j++) {
                float expectedValue = row[j];
                float actualValue = actual.getAt(i, j);
                if (actualValue != expectedValue) {
                    maxErrorPercent = Math.max(
                            maxErrorPercent,
                            100f * Math.abs(actualValue - expectedValue) / (Math.abs(actualValue) + EPSILON));
                }
            }
        }
        return maxErrorPercent;
    }

    @FunctionalInterface
    public interface FloatElementProvider {
        float provide(int row, int column);
    }

}
