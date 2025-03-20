package com.dmytrobilokha.opencl.verification.performance;

import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.Platform;

import java.util.Set;

public interface PerformanceVerifier {

    Set<PerformanceMeasurement> verify(Platform platform, Device device);

    String getName();

}
