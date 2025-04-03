package com.dmytrobilokha.opencl.verification;

public record MatricesMultiplicationSize(int mDimension, int kDimension, int nDimension) {

    @Override
    public String toString() {
        return mDimension + "X" + kDimension + "*" + kDimension + "X" + nDimension;
    }
}
