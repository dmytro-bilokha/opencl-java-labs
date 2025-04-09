package com.dmytrobilokha;

import com.dmytrobilokha.opencl.Platform;

public class ShowInfo {

    public static void main(String[] args) {
        // TODO: consider making CL source non-mandatory to init a platform
        try (var platform = Platform.initDefault(FileUtil.readStringResource("main.cl"))) {
            System.out.println("Default platform name: " + platform.getName());
            System.out.println("Default platform version: " + platform.getVersion());
            var devices = platform.getDevices();
            System.out.println("Number of devices: " + devices.size());
            var device = devices.getFirst();
            System.out.println("Device name: " + device.getName());
            System.out.println("Device version: " + device.getVersion());
            System.out.println("Device C version: " + device.getClangVersion());
            System.out.println("Device global memory size: " + device.getGlobalMemorySize());
            System.out.println("Device local memory size: " + device.getLocalMemorySize());
            System.out.println("Device max compute units: " + device.getMaxComputeUnits());
            System.out.println("Device number of cores: " + device.getNumberOfCores());
            System.out.println("Device max clock frequency: " + device.getMaxClockFrequency());
            System.out.println("Device max work item dimensions: " + device.getMaxWorkItemDimensions());
            System.out.println("Device max work item sizes: " + device.getMaxWorkItemSizes());
            System.out.println("Device max work group size: " + device.getMaxWorkGroupSize());
            System.out.println("Device preferred work group multiple: " + device.getPreferredWorkGroupMultiple());
            System.out.println("Device max memory allocation size: " + device.getMaxMemoryAllocationSize());
            System.out.println("Device max 2D image width: " + device.getMax2dImageWidth());
            System.out.println("Device max 2D image height: " + device.getMax2dImageHeight());
            System.out.println("Device preferred float vector width: " + device.getPreferredVectorWidthFloat());
        }
    }

}
