package com.dmytrobilokha.opencl.operation;

import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.Event;
import com.dmytrobilokha.opencl.Kernel;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.PlatformBuffer;

import java.util.Set;

public class AddMatricesOperation {

    private static final String LEFTOVERS_KERNEL_NAME = "addMatricesLeftover";

    private final Platform platform;
    private final Flavor flavor;
    private final Kernel mainKernel;
    private final Kernel leftoversKernel;
    private long elementsCount;
    private long leftoversCount;

    private AddMatricesOperation(Flavor flavor, Platform platform) {
        this.flavor = flavor;
        this.platform = platform;
        this.mainKernel = platform.createKernel(flavor.kernelName);
        if (flavor.vectorWidth != 1) {
            this.leftoversKernel = platform.createKernel(LEFTOVERS_KERNEL_NAME);
        } else {
            this.leftoversKernel = null;
        }
    }

    public static AddMatricesOperation withFlavor(Flavor flavor, Platform platform) {
        return new AddMatricesOperation(flavor, platform);
    }

    public void setArguments(PlatformBuffer a, PlatformBuffer b, PlatformBuffer result, long rows, long columns) {
        elementsCount = rows * columns;
        platform.setKernelArgument(mainKernel, 0, a);
        platform.setKernelArgument(mainKernel, 1, b);
        platform.setKernelArgument(mainKernel, 2, result);
        platform.setKernelArgument(mainKernel, 3, elementsCount / flavor.vectorWidth);
        leftoversCount = elementsCount % flavor.vectorWidth;
        if (shouldProcessLeftovers()) {
            platform.setKernelArgument(leftoversKernel, 0, a);
            platform.setKernelArgument(leftoversKernel, 1, b);
            platform.setKernelArgument(leftoversKernel, 2, result);
            platform.setKernelArgument(leftoversKernel, 3, elementsCount - leftoversCount);
            platform.setKernelArgument(leftoversKernel, 4, elementsCount);
        }
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        long workSize = Math.min(device.getNumberOfCores(), elementsCount / flavor.vectorWidth);
        var mainKernelEvent = device.enqueueNdRangeKernel(mainKernel, workSize, waitForEvents);
        if (shouldProcessLeftovers()) {
            var leftoversKernelEvent = device.enqueueNdRangeKernel(leftoversKernel, 1, waitForEvents);
            return Set.of(mainKernelEvent, leftoversKernelEvent);
        }
        return Set.of(mainKernelEvent);
    }

    private boolean shouldProcessLeftovers() {
        return leftoversKernel != null && leftoversCount != 0;
    }

    public enum Flavor {
        FLOAT1("addMatrices", 1),
        FLOAT2("addMatrices2", 2),
        FLOAT4("addMatrices4", 4),
        FLOAT8("addMatrices8", 8),
        FLOAT16("addMatrices16", 16),
        ;

        final String kernelName;
        final int vectorWidth;

        Flavor(String kernelName, int vectorWidth) {
            this.kernelName = kernelName;
            this.vectorWidth = vectorWidth;
        }
    }

}
