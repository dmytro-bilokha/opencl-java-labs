package com.dmytrobilokha.opencl.verification;

public record MatrixSize(int rows, int columns) {

    public int numberOfElements() {
        return rows * columns;
    }

    @Override
    public String toString() {
        return rows + "X" + columns;
    }
}
