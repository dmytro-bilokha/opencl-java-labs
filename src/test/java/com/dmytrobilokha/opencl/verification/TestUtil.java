package com.dmytrobilokha.opencl.verification;

import com.dmytrobilokha.memory.FloatMemoryMatrix;
import org.testng.Assert;

public class TestUtil {

    private TestUtil() {
        // no instance
    }

    public static void assertMatricesEqual(FloatMemoryMatrix actual, FloatMatrix expected) {
        float[][] verificationData = expected.getData();
        for (int i = 0; i < expected.getRowDimension(); i++) {
            float[] row = verificationData[i];
            for (int j = 0; j < expected.getColumnDimension(); j++) {
                float expectedValue = row[j];
                float actualValue = actual.getAt(i, j);
                Assert.assertEquals(
                        actualValue,
                        expectedValue,
                        "For (" + i + ", " + j + ") expected " + expectedValue + ", but got " + actualValue);
            }
        }
    }

}
