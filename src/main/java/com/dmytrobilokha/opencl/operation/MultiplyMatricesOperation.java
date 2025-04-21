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
        platform.setKernelArgument(mainKernel, 3, mDimension);
        platform.setKernelArgument(mainKernel, 4, kDimension);
        platform.setKernelArgument(mainKernel, 5, nDimension);
        globalWorkSize = new long[]{mDimension / flavor.vectorWidth, nDimension};
        localWorkSize = flavor.tileSize == null ? null : new long[]{flavor.tileSize / flavor.vectorWidth, flavor.tileSize};
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        Event mainKernelEvent = device.enqueueNdRangeKernel(mainKernel, localWorkSize, globalWorkSize, waitForEvents);
        return Set.of(mainKernelEvent);
    }

    public enum Flavor {
        FLOAT_SIMPLE("multiplyMatricesSimple", 32, 1),
        FLOAT_TILE_32("multiplyMatricesTile32", 32, 1),
        FLOAT_TILE_16("multiplyMatricesTile16", 16, 1),
        FLOAT_TILE_32W8("multiplyMatricesTile32W8", 32, 8),
        FLOAT_TILE_32W4("multiplyMatricesTile32W4", 32, 4),
        FLOAT_TILE_16W4("multiplyMatricesTile16W4", 16, 4),
        FLOAT_TILE_32V4("multiplyMatricesTile32V4", 32, 4),
        FLOAT_TILE_32V8("multiplyMatricesTile32V8", 32, 8),
        ;

        final String kernelName;
        final Integer tileSize;
        final int vectorWidth;

        Flavor(String kernelName, Integer tileSize, int vectorWidth) {
            this.kernelName = kernelName;
            this.tileSize = tileSize;
            this.vectorWidth = vectorWidth;
        }
    }

}
