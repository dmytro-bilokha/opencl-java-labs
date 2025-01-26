package com.dmytrobilokha.opencl;

import com.dmytrobilokha.opencl.binding.ParamValue;

public enum HostMemoryAccess {

    READ_ONLY(ParamValue.CL_MEM_HOST_READ_ONLY),
    WRITE_ONLY(ParamValue.CL_MEM_HOST_WRITE_ONLY),
    READ_WRITE(0L), // This is default, no flags required
    NO_ACCESS(ParamValue.CL_MEM_HOST_NO_ACCESS);

    private final long paramValue;

    HostMemoryAccess(long paramValue) {
        this.paramValue = paramValue;
    }

    long getParamValue() {
        return paramValue;
    }

}
