package com.dmytrobilokha.opencl.verification;

import com.dmytrobilokha.FileUtil;
import com.dmytrobilokha.opencl.Device;
import com.dmytrobilokha.opencl.Platform;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

@Test(groups = {"verification"})
public class PlatformHandler {

    public static Platform platform;
    public static Device device;

    @BeforeSuite
    void init() {
        platform = Platform.initDefault(FileUtil.readStringResource("main.cl"));
        device = platform.getDevices().getFirst();
    }

    @AfterSuite
    void shutdown() {
        platform.close();
    }

}
