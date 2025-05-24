package com.dmytrobilokha.opencl.operation;

import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.Event;
import com.dmytrobilokha.opencl.Kernel;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.PlatformBuffer;

import java.util.Set;

public class PaddingOperation {

    private final Flavor flavor;
    private final Platform platform;
    private final Kernel kernel;
    private long[] localWorkSize;
    private long[] globalWorkSize;

    private PaddingOperation(Flavor flavor, Platform platform) {
        this.flavor = flavor;
        this.platform = platform;
        this.kernel = platform.createKernel(flavor.kernelName);
    }

    public static PaddingOperation withFlavor(Flavor flavor, Platform platform) {
        return new PaddingOperation(flavor, platform);
    }

    public void setArguments(
            PlatformBuffer input,
            PlatformBuffer output,
            long inputRows,
            long inputColumns,
            long outputRows,
            long outputColumns) {
        platform.setKernelArgument(kernel, 0, input);
        platform.setKernelArgument(kernel, 1, output);
        platform.setKernelArgument(kernel, 2, flavor.tileSize);
        platform.setKernelArgument(kernel, 3, inputRows);
        platform.setKernelArgument(kernel, 4, inputColumns);
        platform.setKernelArgument(kernel, 5, outputRows);
        platform.setKernelArgument(kernel, 6, outputColumns);
        globalWorkSize = new long[]{outputColumns, outputRows};
        localWorkSize = new long[]{flavor.tileSize, flavor.tileSize};
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        Event event = device.enqueueNdRangeKernel(kernel, localWorkSize, globalWorkSize, waitForEvents);
        return Set.of(event);
    }

    public enum Flavor {
        FLOAT_16("padRightBottom", 16),
        FLOAT_32("padRightBottom", 32),
        FLOAT_64("padRightBottom", 64),
        ;

        final String kernelName;
        final int tileSize;

        Flavor(String kernelName, int tileSize) {
            this.kernelName = kernelName;
            this.tileSize = tileSize;
        }

        public int getTileSize() {
            return tileSize;
        }
    }

}
