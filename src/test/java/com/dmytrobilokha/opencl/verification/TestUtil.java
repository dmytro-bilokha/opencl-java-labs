package com.dmytrobilokha.opencl.verification;

import com.dmytrobilokha.memory.FloatMemoryMatrix;
import com.dmytrobilokha.opencl.ProfilingInfo;
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

    public static void printEventProfiling(ProfilingInfo profilingInfo, String name) {
        System.out.println(name + " event profiling, microseconds:");
        System.out.println("    submitted - queued = " + (profilingInfo.commandSubmittedNanos() - profilingInfo.commandQueuedNanos()) / 1000);
        System.out.println("    started - submitted = " + (profilingInfo.commandStartedNanos() - profilingInfo.commandSubmittedNanos()) / 1000);
        System.out.println("    finished - started = " + (profilingInfo.commandFinishedNanos() - profilingInfo.commandStartedNanos()) / 1000);
        System.out.println("    completed - finished = " + (profilingInfo.commandCompletedNanos() - profilingInfo.commandFinishedNanos()) / 1000);
    }

}
