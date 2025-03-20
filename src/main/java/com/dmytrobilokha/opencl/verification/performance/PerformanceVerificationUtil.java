package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.opencl.ProfilingInfo;

import java.util.Collection;

public final class PerformanceVerificationUtil {

    private PerformanceVerificationUtil() {
        // no instance
    }

    public static long calculateTotalDuration(Collection<ProfilingInfo> profilingInfos) {
        long minStartTime = profilingInfos
                .stream()
                .mapToLong(ProfilingInfo::commandStartedNanos)
                .min()
                .getAsLong();
        long maxFinishTime = profilingInfos
                .stream()
                .mapToLong(ProfilingInfo::commandFinishedNanos)
                .max()
                .getAsLong();
        return maxFinishTime - minStartTime;
    }

}
