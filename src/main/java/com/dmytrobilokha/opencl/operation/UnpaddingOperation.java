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
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        Event event = device.enqueueNdRangeKernel(kernel, null, globalWorkSize, waitForEvents);
        return Set.of(event);
    }

    public enum Flavor {
        FLOAT("unpadRightBottom"),
        ;

        final String kernelName;

        Flavor(String kernelName) {
            this.kernelName = kernelName;
        }
    }
    
}
