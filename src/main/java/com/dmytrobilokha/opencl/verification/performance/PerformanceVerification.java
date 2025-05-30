package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.FileUtil;
import com.dmytrobilokha.opencl.Platform;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PerformanceVerification {

    private static final long SHOW_LIMIT = 3;

    private static final List<PerformanceVerifier> ALL_VERIFIERS = List.of(
            new AddMatricesPerformanceVerifier(),
            new SigmoidElementsPerformanceVerifier(),
            new MultiplyMatricesPerformanceVerifier()
    );

    public static void main(String[] args) {
        boolean verbose = Arrays.stream(args)
                .anyMatch(arg -> "-v".equals(arg) || "--verbose".equals(arg));
        var verifierFilters = Arrays.stream(args)
                .filter(arg -> !arg.startsWith("-"))
                .collect(Collectors.toSet());
        List<PerformanceVerifier> verifiers;
        if (verifierFilters.isEmpty()) {
            verifiers = ALL_VERIFIERS;
        } else {
            verifiers = new ArrayList<>();
            for (String verifierFilter : verifierFilters) {
                ALL_VERIFIERS
                        .stream()
                        .filter(verifier ->
                                verifier.getClass()
                                        .getSimpleName()
                                        .toUpperCase()
                                        .startsWith(verifierFilter.toUpperCase())
                        )
                        .forEach(verifiers::add);
            }
        }
        PrintWriter reportWriter = new PrintWriter(System.out);
        if (verifiers.isEmpty()) {
            reportWriter.println("No verifiers found for filters: " + String.join(",", verifierFilters));
        }
        try (var platform = Platform.initDefault(FileUtil.readStringResource("main.cl"))) {
            var device = platform.getDevices().getFirst();
            for (var verifier : verifiers) {
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
                // in non-verbose mode, show top performers ordered by performance and failures at the end
                measurements = Stream.concat(
                        measurementsByDescription
                                .get(description)
                                .stream()
                                .sorted(Comparator.comparing(PerformanceMeasurement::flops).reversed())
                                .limit(SHOW_LIMIT),
                        measurementsByDescription
                                .get(description)
                                .stream()
                                .filter(pm -> pm.exception() != null)
                ).toList();
            }
            for (var measurement : measurements) {
                if (measurement.exception() == null) {
                    reportWriter.println(
                            name + " " + description + " "
                                    + measurement.flavor() + ": " + formatFlops(measurement.flops())
                                    + formatRemark(measurement.remark())
                    );
                } else {
                    reportWriter.println(
                            name + " " + description + " "
                                    + measurement.flavor() + ": " + "FAILURE"
                                    + formatRemark(measurement.remark())
                    );
                    if (verbose) {
                        measurement.exception().printStackTrace(reportWriter);
                    }
                }
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
