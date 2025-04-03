package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.FileUtil;
import com.dmytrobilokha.opencl.Platform;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Arrays;

public class CorrectnessVerification {

    private static final CorrectnessVerifier[] VERIFIERS = new CorrectnessVerifier[]{
            new AddMatricesCorrectnessVerifier(),
            new SigmoidElementsCorrectnessVerifier(),
            new MultiplyMatricesCorrectnessVerifier(),
    };

    public static void main(String[] args) {
        boolean verbose = Arrays.stream(args)
                .anyMatch(arg -> "-v".equals(arg) || "--verbose".equals(arg));
        PrintWriter reportWriter = new PrintWriter(System.out);
        StringWriter checkReportWriter = null;
        try (var platform = Platform.initDefault(FileUtil.readStringResource("main.cl"))) {
            var device = platform.getDevices().getFirst();
            for (var verifier : VERIFIERS) {
                checkReportWriter = new StringWriter();
                boolean isOk = verifier.verify(platform, device, new PrintWriter(checkReportWriter));
                if (isOk) {
                    reportWriter.println(verifier.getClass().getSimpleName() + " all checks PASSED");
                    if (verbose) {
                        reportWriter.print(checkReportWriter);
                    }
                } else {
                    reportWriter.println(verifier.getClass().getSimpleName() + " detected FAILURE!");
                    reportWriter.print(checkReportWriter);
                }
            }
        } catch (RuntimeException e) {
            reportWriter.println("Unexpected exception while running correctness checks: ");
            if (checkReportWriter != null) {
                reportWriter.print(checkReportWriter);
            }
            e.printStackTrace(reportWriter);
        } finally {
            reportWriter.flush();
            reportWriter.close();
        }
    }

}
