package com.dmytrobilokha.opencl;

import java.lang.foreign.MemorySegment;

public class Kernel {

    private final String functionName;
    private final MemorySegment kernelMemSeg;

    public Kernel(String functionName, MemorySegment kernelMemSeg) {
        this.functionName = functionName;
        this.kernelMemSeg = kernelMemSeg;
    }

    MemorySegment getKernelMemSeg() {
        return kernelMemSeg;
    }
}
