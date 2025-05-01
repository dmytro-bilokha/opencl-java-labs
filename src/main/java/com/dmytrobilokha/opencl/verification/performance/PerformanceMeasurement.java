package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.opencl.exception.OpenClRuntimeException;

public record PerformanceMeasurement(
        String description, String flavor, long flops, String remark, OpenClRuntimeException exception) {

    public static PerformanceMeasurement ofSuccess(String description, String flavor, long flops, String remark) {
        return new PerformanceMeasurement(description, flavor, flops, remark, null);
    }

    public static PerformanceMeasurement ofSuccess(String description, String flavor, long flops) {
        return new PerformanceMeasurement(description, flavor, flops, "", null);
    }

    public static PerformanceMeasurement ofFailure(String description, String flavor, OpenClRuntimeException exception) {
        String remark = exception.getClErrorCode() == null
                ? exception.getMessage()
                : exception.getClErrorCode().name();
        return new PerformanceMeasurement(description, flavor, 0, remark, exception);
    }

    public PerformanceMeasurement(String description, String flavor, long flops) {
        this(description, flavor, flops, "", null);
    }

}
