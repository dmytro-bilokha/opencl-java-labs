package com.dmytrobilokha.opencl;

public enum HostMemoryAccess {

    READ_ONLY(ClParamValue.CL_MEM_HOST_READ_ONLY),
    WRITE_ONLY(ClParamValue.CL_MEM_HOST_WRITE_ONLY),
    READ_WRITE(0L), // This is default, no flags required
    NO_ACCESS(ClParamValue.CL_MEM_HOST_NO_ACCESS);

    private final long paramValue;

    HostMemoryAccess(long paramValue) {
        this.paramValue = paramValue;
    }

    public long getParamValue() {
        return paramValue;
    }

}
