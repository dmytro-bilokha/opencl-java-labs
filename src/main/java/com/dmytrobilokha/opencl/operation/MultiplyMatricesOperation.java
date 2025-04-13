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
    private long[] localWorkSize;
    private long[] globalWorkSize;

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
        if (flavor == Flavor.FLOAT_TILE_32MI || flavor == Flavor.FLOAT_TILE_32I) {
            platform.setKernelArgument(mainKernel, 3, (int) mDimension);
            platform.setKernelArgument(mainKernel, 4, (int) kDimension);
            platform.setKernelArgument(mainKernel, 5, (int) nDimension);
        } else {
            platform.setKernelArgument(mainKernel, 3, mDimension);
            platform.setKernelArgument(mainKernel, 4, kDimension);
            platform.setKernelArgument(mainKernel, 5, nDimension);
        }
        switch (flavor) {
            case FLOAT_N -> {
                globalWorkSize = new long[]{mDimension, nDimension};
                localWorkSize = new long[]{32L, 32L};
            }
            case FLOAT_NO -> {
                globalWorkSize = new long[]{nDimension, mDimension};
                localWorkSize = new long[]{32L, 32L};
            }
            case FLOAT_TILE_32, FLOAT_TILE_32I -> {
                globalWorkSize = new long[]{nDimension, mDimension};
                localWorkSize = new long[]{32L, 32L};
            }
            case FLOAT_TILE_32M, FLOAT_TILE_32MI -> {
                globalWorkSize = new long[]{nDimension / 8, mDimension};
                localWorkSize = new long[]{32L / 8L, 32L};
            }
        }
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        Event mainKernelEvent = device.enqueueNdRangeKernel(mainKernel, localWorkSize, globalWorkSize, waitForEvents);
        return Set.of(mainKernelEvent);
    }

    public enum Flavor {
        FLOAT_N("multiplyMatricesNaive", 1),
        FLOAT_NO("multiplyMatricesNaiveO", 1),
        FLOAT_TILE_32("multiplyMatricesTile32", 1),
        FLOAT_TILE_32I("multiplyMatricesTile32i", 1),
        FLOAT_TILE_32M("multiplyMatricesTile32M", 1),
        FLOAT_TILE_32MI("multiplyMatricesTile32Mi", 1),
        ;

        final String kernelName;
        final int vectorWidth;

        Flavor(String kernelName, int vectorWidth) {
            this.kernelName = kernelName;
            this.vectorWidth = vectorWidth;
        }
    }

}
