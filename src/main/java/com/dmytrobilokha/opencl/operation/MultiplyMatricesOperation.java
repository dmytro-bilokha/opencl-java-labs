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
        globalWorkSize = new long[]{mDimension / flavor.vectorWidth, nDimension / flavor.workHeight};
        localWorkSize = flavor.tileSize == null
                ? null
                : new long[]{flavor.tileSize / flavor.vectorWidth, flavor.tileSize / flavor.workHeight};
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        Event mainKernelEvent = device.enqueueNdRangeKernel(mainKernel, localWorkSize, globalWorkSize, waitForEvents);
        return Set.of(mainKernelEvent);
    }

    public enum Flavor {
        FLOAT_SIMPLE("multiplyMatricesSimple", 32, 1, 1),
        FLOAT_TILE_32("multiplyMatricesTile32", 32, 1, 1),
        FLOAT_TILE_16("multiplyMatricesTile16", 16, 1, 1),
        FLOAT_TILE_32W8("multiplyMatricesTile32W8", 32, 8, 1),
        FLOAT_TILE_32W4("multiplyMatricesTile32W4", 32, 4, 1),
        FLOAT_TILE_16W4("multiplyMatricesTile16W4", 16, 4, 1),
        FLOAT_TILE_32V4("multiplyMatricesTile32V4", 32, 4, 1),
        FLOAT_TILE_32V8("multiplyMatricesTile32V8", 32, 8, 1),
        FLOAT_TILE_32V8H2("multiplyMatricesTile32V8H2", 32, 8, 2),
        FLOAT_TILE_32V8H8P("multiplyMatricesTile32V8H8P", 32, 8, 8),
        FLOAT_TILE_32V4H2("multiplyMatricesTile32V4H2", 32, 4, 2),
        FLOAT_TILE_32V4H2P("multiplyMatricesTile32V4H2P", 32, 4, 2),
        FLOAT_TILE_64V4H2("multiplyMatricesTile64V4H2", 64, 4, 2),
        FLOAT_TILE_64V4H4("multiplyMatricesTile64V4H4", 64, 4, 4),
        FLOAT_TILE_64V4H8("multiplyMatricesTile64V4H8", 64, 4, 8),
        FLOAT_TILE_64V4H8P("multiplyMatricesTile64V4H8P", 64, 4, 8),
        FLOAT_TILE_64V4H16("multiplyMatricesTile64V4H16", 64, 4, 16),
        FLOAT_TILE_64V8("multiplyMatricesTile64V8", 64, 8, 1),
        FLOAT_TILE_64V8H2("multiplyMatricesTile64V8H2", 64, 8, 2),
        FLOAT_TILE_32V4H4("multiplyMatricesTile32V4H4", 32, 4, 4),
        FLOAT_TILE_32V4H8("multiplyMatricesTile32V4H8", 32, 4, 8),
        FLOAT_TILE_32V4H8P("multiplyMatricesTile32V4H8P", 32, 4, 8),
        ;

        final String kernelName;
        final Integer tileSize;
        final int vectorWidth;
        final int workHeight;

        Flavor(String kernelName, Integer tileSize, int vectorWidth, int workHeight) {
            this.kernelName = kernelName;
            this.tileSize = tileSize;
            this.vectorWidth = vectorWidth;
            this.workHeight = workHeight;
        }
    }

}
