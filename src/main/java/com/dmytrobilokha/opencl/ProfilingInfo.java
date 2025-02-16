package com.dmytrobilokha.opencl;

public record ProfilingInfo(
        long commandQueuedNanos,
        long commandSubmittedNanos,
        long commandStartedNanos,
        long commandFinishedNanos,
        long commandCompletedNanos)
 {}
