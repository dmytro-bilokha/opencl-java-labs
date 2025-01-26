package com.dmytrobilokha.opencl;

public enum DeviceMemoryAccess {
    READ_ONLY(ClParamValue.CL_MEM_READ_ONLY),
    WRITE_ONLY(ClParamValue.CL_MEM_WRITE_ONLY),
    READ_WRITE(ClParamValue.CL_MEM_READ_WRITE);

    private final long paramValue;

    DeviceMemoryAccess(long paramValue) {
        this.paramValue = paramValue;
    }

    public long getParamValue() {
        return paramValue;
    }

}
