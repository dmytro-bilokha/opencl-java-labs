package com.dmytrobilokha.opencl;

import com.dmytrobilokha.opencl.binding.ParamValue;

public enum DeviceMemoryAccess {
    READ_ONLY(ParamValue.CL_MEM_READ_ONLY),
    WRITE_ONLY(ParamValue.CL_MEM_WRITE_ONLY),
    READ_WRITE(ParamValue.CL_MEM_READ_WRITE);

    private final long paramValue;

    DeviceMemoryAccess(long paramValue) {
        this.paramValue = paramValue;
    }

    long getParamValue() {
        return paramValue;
    }

}
