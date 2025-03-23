package com.dmytrobilokha.opencl.operation;

import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.Event;
import com.dmytrobilokha.opencl.Kernel;
import com.dmytrobilokha.opencl.Platform;
import com.dmytrobilokha.opencl.PlatformBuffer;

import java.util.Set;

public class SigmoidElementsOperation {

    private final Platform platform;
    private final Flavor flavor;
    private final Kernel mainKernel;
    private final Kernel leftoverKernel;
    private long elementsCount;
    private long leftoversCount;

    private SigmoidElementsOperation(Flavor flavor, Platform platform) {
        this.flavor = flavor;
        this.platform = platform;
        this.mainKernel = platform.createKernel(flavor.kernelName);
        if (flavor.vectorWidth != 1) {
            this.leftoverKernel = platform.createKernel(flavor.leftoverKernelName);
        } else {
            this.leftoverKernel = null;
        }
    }

    public static SigmoidElementsOperation withFlavor(Flavor flavor, Platform platform) {
        return new SigmoidElementsOperation(flavor, platform);
    }

    public void setArguments(PlatformBuffer input, PlatformBuffer output, long rows, long columns) {
        elementsCount = rows * columns;
        platform.setKernelArgument(mainKernel, 0, input);
        platform.setKernelArgument(mainKernel, 1, output);
        platform.setKernelArgument(mainKernel, 2, elementsCount / flavor.vectorWidth);
        leftoversCount = elementsCount % flavor.vectorWidth;
        if (shouldProcessLeftovers()) {
            platform.setKernelArgument(leftoverKernel, 0, input);
            platform.setKernelArgument(leftoverKernel, 1, output);
            platform.setKernelArgument(leftoverKernel, 2, elementsCount - leftoversCount);
            platform.setKernelArgument(leftoverKernel, 3, elementsCount);
        }
    }

    public Set<Event> enqueue(Device device, Set<Event> waitForEvents) {
        long workSize = Math.min(device.getNumberOfCores(), elementsCount / flavor.vectorWidth);
        var mainKernelEvent = device.enqueueNdRangeKernel(mainKernel, workSize, waitForEvents);
        if (shouldProcessLeftovers()) {
            var leftoversKernelEvent = device.enqueueNdRangeKernel(leftoverKernel, 1, waitForEvents);
            return Set.of(mainKernelEvent, leftoversKernelEvent);
        }
        return Set.of(mainKernelEvent);
    }

    private boolean shouldProcessLeftovers() {
        return leftoverKernel != null && leftoversCount != 0;
    }

    public enum Flavor {
        FLOAT1("sigmoidElements", "sigmoidElementsLeftover", 1),
        FLOAT2("sigmoidElements2", "sigmoidElementsLeftover", 2),
        FLOAT4("sigmoidElements4", "sigmoidElementsLeftover", 4),
        FLOAT8("sigmoidElements8", "sigmoidElementsLeftover", 8),
        FLOAT16("sigmoidElements16", "sigmoidElementsLeftover", 16),
        FLOAT1_NATIVE("sigmoidElementsN", "sigmoidElementsLeftoverN", 1),
        FLOAT2_NATIVE("sigmoidElements2N", "sigmoidElementsLeftoverN", 2),
        FLOAT4_NATIVE("sigmoidElements4N", "sigmoidElementsLeftoverN", 4),
        FLOAT8_NATIVE("sigmoidElements8N", "sigmoidElementsLeftoverN", 8),
        FLOAT16_NATIVE("sigmoidElements16N", "sigmoidElementsLeftoverN", 16),
        ;

        final String kernelName;
        final String leftoverKernelName;
        final int vectorWidth;

        Flavor(String kernelName, String leftoverKernelName, int vectorWidth) {
            this.kernelName = kernelName;
            this.leftoverKernelName = leftoverKernelName;
            this.vectorWidth = vectorWidth;
        }
    }

}
