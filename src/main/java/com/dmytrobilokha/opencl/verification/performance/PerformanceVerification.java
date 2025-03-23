package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.FileUtil;
import com.dmytrobilokha.opencl.Platform;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class PerformanceVerification {

    private static final long SHOW_LIMIT = 3;

    private static final PerformanceVerifier[] VERIFIERS = new PerformanceVerifier[]{
            new AddMatricesOperationPerformanceVerifier(),
            new SigmoidElementsOperationPerformanceVerifier(),
    };

    public static void main(String[] args) {
        boolean verbose = Arrays.stream(args)
                .anyMatch(arg -> "-v".equals(arg) || "--verbose".equals(arg));
        PrintWriter reportWriter = new PrintWriter(System.out);
        try (var platform = Platform.initDefault(FileUtil.readStringResource("main.cl"))) {
            var device = platform.getDevices().getFirst();
            for (var verifier : VERIFIERS) {
                var performanceMeasurements = verifier.verify(platform, device);
                reportPerformanceMeasurements(verifier.getName(), performanceMeasurements, reportWriter, verbose);
            }
        } catch (RuntimeException e) {
            reportWriter.println("Unexpected exception while running performance checks: ");
            e.printStackTrace(reportWriter);
        } finally {
            reportWriter.flush();
            reportWriter.close();
        }
    }

    private static void reportPerformanceMeasurements(
            String name,
            Set<PerformanceMeasurement> performanceMeasurements,
            PrintWriter reportWriter,
            boolean verbose
    ) {
        var measurementsByDescription = performanceMeasurements
                .stream()
                .collect(Collectors.groupingBy(PerformanceMeasurement::description));
        var sortedDescriptions = performanceMeasurements
                .stream()
                .map(PerformanceMeasurement::description)
                .distinct()
                .sorted()
                .toList();
        for (String description : sortedDescriptions) {
            List<PerformanceMeasurement> measurements;
            if (verbose) {
                // in verbose mode, show all measurements sorted by flavor
                measurements = measurementsByDescription
                        .get(description)
                        .stream()
                        .sorted(Comparator.comparing(PerformanceMeasurement::flavor))
                        .toList();
            } else {
                // in non-verbose mode, show top performers ordered by performance
                measurements = measurementsByDescription
                        .get(description)
                        .stream()
                        .sorted(Comparator.comparing(PerformanceMeasurement::flops).reversed())
                        .limit(SHOW_LIMIT)
                        .toList();
            }
            for (var measurement : measurements) {
                reportWriter.println(
                        name + " " + description + " "
                                + measurement.flavor() + ": " + formatFlops(measurement.flops())
                        + formatRemark(measurement.remark())
                );
            }
        }
    }

    private static String formatRemark(String remark) {
        return remark.isEmpty()
                ? ""
                : (" (" + remark + ")");
    }

    private static String formatFlops(long flops) {
        if (flops > 10_000_000_000L) {
            return flops / 1_000_000_000 + " GFLOPS";
        }
        return flops / 1_000_000 + " MFLOPS";
    }

}
