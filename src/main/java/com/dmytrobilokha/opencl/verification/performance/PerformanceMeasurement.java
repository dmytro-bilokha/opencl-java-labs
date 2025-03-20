package com.dmytrobilokha.opencl.verification.performance;

public record PerformanceMeasurement(String description, String flavor, long flops, String remark) {

    public PerformanceMeasurement(String description, String flavor, long flops) {
        this(description, flavor, flops, "");
    }

}
