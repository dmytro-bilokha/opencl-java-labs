package com.dmytrobilokha.opencl.operation;

import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.Event;
import com.dmytrobilokha.opencl.Kernel;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.PlatformBuffer;

import java.util.Set;

public class UnpaddingOperation {

    private final Flavor flavor;
    private final Platform platform;
    private final Kernel kernel;
    private long[] localWorkSize;
    private long[] globalWorkSize;

    private UnpaddingOperation(Flavor flavor, Platform platform) {
        this.flavor = flavor;
        this.platform = platform;
        this.kernel = platform.createKernel(flavor.kernelName);
    }

    public static UnpaddingOperation withFlavor(Flavor flavor, Platform platform) {
        return new UnpaddingOperation(flavor, platform);
    }

    public void setArguments(
            PlatformBuffer input,
            PlatformBuffer output,
            long inputColumns,
            long outputRows,
            long outputColumns) {
        platform.setKernelArgument(kernel, 0, input);
        platform.setKernelArgument(kernel, 1, output);
        platform.setKernelArgument(kernel, 2, inputColumns);
        platform.setKernelArgument(kernel, 3, outputColumns);
        globalWorkSize = new long[]{outputColumns, outputRows};
        localWorkSize = new long[]{flavor.padSize, flavor.padSize};
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        Event event = device.enqueueNdRangeKernel(kernel, localWorkSize, globalWorkSize, waitForEvents);
        return Set.of(event);
    }

    public enum Flavor {
        FLOAT_16("unpadRightBottom", 16),
        FLOAT_32("unpadRightBottom", 32),
        FLOAT_64("unpadRightBottom", 64),
        ;

        final String kernelName;
        final int padSize;

        Flavor(String kernelName, int padSize) {
            this.kernelName = kernelName;
            this.padSize = padSize;
        }
    }
    
}
