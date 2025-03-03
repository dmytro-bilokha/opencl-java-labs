package com.dmytrobilokha.opencl;

import java.lang.foreign.MemorySegment;

public class PlatformBuffer {

    private final long byteSize;
    private final MemorySegment bufferMemSeg;
    private final DeviceMemoryAccess deviceMemoryAccess;
    private final HostMemoryAccess hostMemoryAccess;

    public PlatformBuffer(long byteSize, MemorySegment bufferMemSeg, DeviceMemoryAccess deviceMemoryAccess, HostMemoryAccess hostMemoryAccess) {
        this.byteSize = byteSize;
        this.bufferMemSeg = bufferMemSeg;
        this.deviceMemoryAccess = deviceMemoryAccess;
        this.hostMemoryAccess = hostMemoryAccess;
    }

    public long getByteSize() {
        return byteSize;
    }

    MemorySegment getBufferMemSeg() {
        return bufferMemSeg;
    }

    public DeviceMemoryAccess getDeviceMemoryAccess() {
        return deviceMemoryAccess;
    }

    public HostMemoryAccess getHostMemoryAccess() {
        return hostMemoryAccess;
    }

    @Override
    public String toString() {
        return "ClBuffer{" +
                "byteSize=" + byteSize +
                ", bufferMemSeg=" + bufferMemSeg +
                ", deviceMemoryAccess=" + deviceMemoryAccess +
                ", hostMemoryAccess=" + hostMemoryAccess +
                '}';
    }
}
