package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.opencl.ProfilingInfo;

import java.math.BigInteger;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

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

    public static long calculateFlops(long numberOfOperations, List<ProfilingInfo> profilingInfos) {
        return calculateFlops(numberOfOperations, calculateTotalDuration(profilingInfos));
    }

    public static long calculateFlops(long numberOfOperations, long nanoDuration) {
        return BigInteger.valueOf(numberOfOperations)
                .multiply(BigInteger.valueOf(1_000_000_000))
                .divide(BigInteger.valueOf(nanoDuration))
                .longValueExact();
    }

    public static PerformanceMeasurement createMeasurement(
            String description,
            String flavor,
            long numberOfOperations,
            List<ProfilingInfo> profilingInfos
    ) {
        return new PerformanceMeasurement(
                description,
                flavor,
                calculateFlops(numberOfOperations, profilingInfos),
                buildProfilingRemark(profilingInfos)
        );
    }

    private static String buildProfilingRemark(List<ProfilingInfo> profilingInfos) {
        long minStartTime = profilingInfos
                .stream()
                .mapToLong(ProfilingInfo::commandStartedNanos)
                .min()
                .getAsLong();
        return profilingInfos
                .stream()
                .map(pi ->
                        (pi.commandStartedNanos() - minStartTime)
                                + "-"
                                + (pi.commandFinishedNanos() - minStartTime))
                .collect(Collectors.joining(","));
    }
}
