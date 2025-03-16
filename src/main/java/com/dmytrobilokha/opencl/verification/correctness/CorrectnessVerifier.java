package com.dmytrobilokha.opencl.verification.correctness;

import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.Platform;

import java.io.PrintWriter;

public interface CorrectnessVerifier {

    boolean verify(Platform platform, Device device, PrintWriter reportWriter);

}
