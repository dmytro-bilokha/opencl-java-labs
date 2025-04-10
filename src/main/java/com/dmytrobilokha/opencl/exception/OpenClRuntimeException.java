package com.dmytrobilokha.opencl.exception;

import com.dmytrobilokha.opencl.binding.ReturnValue;

public class OpenClRuntimeException extends RuntimeException {

    private final ReturnValue clErrorCode;

    public OpenClRuntimeException(String message) {
        super(message);
        this.clErrorCode = null;
    }

    public OpenClRuntimeException(String message, ReturnValue clErrorCode) {
        super(message);
        this.clErrorCode = clErrorCode;
    }

    public OpenClRuntimeException(String message, Throwable throwable) {
        super(message, throwable);
        this.clErrorCode = null;
    }

    public OpenClRuntimeException(String message, ReturnValue clErrorCode, Throwable throwable) {
        super(message, throwable);
        this.clErrorCode = clErrorCode;
    }

    public ReturnValue getClErrorCode() {
        return clErrorCode;
    }

}
