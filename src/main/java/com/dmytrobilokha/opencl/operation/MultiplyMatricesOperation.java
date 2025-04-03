package com.dmytrobilokha.opencl.operation;

import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.Event;
import com.dmytrobilokha.opencl.Kernel;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.PlatformBuffer;

import java.util.Set;

public class MultiplyMatricesOperation {

    private final Platform platform;
    private final Flavor flavor;
    private final Kernel mainKernel;
    private long[] workSize;

    private MultiplyMatricesOperation(Flavor flavor, Platform platform) {
        this.flavor = flavor;
        this.platform = platform;
        this.mainKernel = platform.createKernel(flavor.kernelName);
    }

    public static MultiplyMatricesOperation withFlavor(Flavor flavor, Platform platform) {
        return new MultiplyMatricesOperation(flavor, platform);
    }

    public void setArguments(
            PlatformBuffer a,
            PlatformBuffer b,
            PlatformBuffer result,
            long mDimension,
            long kDimension,
            long nDimension) {
        platform.setKernelArgument(mainKernel, 0, a);
        platform.setKernelArgument(mainKernel, 1, b);
        platform.setKernelArgument(mainKernel, 2, result);
        platform.setKernelArgument(mainKernel, 3, mDimension);
        platform.setKernelArgument(mainKernel, 4, kDimension);
        platform.setKernelArgument(mainKernel, 5, nDimension);
        if (flavor == Flavor.FLOAT_N) {
            workSize = new long[]{mDimension, nDimension};
        } else if (flavor == Flavor.FLOAT_NO) {
            workSize = new long[]{nDimension, mDimension};
        }
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        var mainKernelEvent = device.enqueueNdRangeKernel(mainKernel, workSize, waitForEvents);
        return Set.of(mainKernelEvent);
    }

    public enum Flavor {
        FLOAT_N("multiplyMatricesNaive", 1),
        FLOAT_NO("multiplyMatricesNaiveO", 1),
        ;

        final String kernelName;
        final int vectorWidth;

        Flavor(String kernelName, int vectorWidth) {
            this.kernelName = kernelName;
            this.vectorWidth = vectorWidth;
        }
    }

}
