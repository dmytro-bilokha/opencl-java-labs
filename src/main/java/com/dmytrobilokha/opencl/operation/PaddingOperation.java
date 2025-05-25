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
        platform.setKernelArgument(kernel, 2, inputRows);
        platform.setKernelArgument(kernel, 3, inputColumns);
        platform.setKernelArgument(kernel, 4, outputRows);
        platform.setKernelArgument(kernel, 5, outputColumns);
        globalWorkSize = new long[]{outputColumns, outputRows};
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        Event event = device.enqueueNdRangeKernel(kernel, null, globalWorkSize, waitForEvents);
        return Set.of(event);
    }

    public enum Flavor {
        FLOAT("padRightBottom"),
        ;

        final String kernelName;

        Flavor(String kernelName) {
            this.kernelName = kernelName;
        }

    }

}
