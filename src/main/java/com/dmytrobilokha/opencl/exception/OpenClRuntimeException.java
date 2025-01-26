package com.dmytrobilokha.opencl.exception;

public class OpenClRuntimeException extends RuntimeException {

    public OpenClRuntimeException(String message) {
        super(message);
    }

    public OpenClRuntimeException(String message, Throwable throwable) {
        super(message, throwable);
    }

}
