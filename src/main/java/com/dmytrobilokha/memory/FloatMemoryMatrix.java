package com.dmytrobilokha.memory;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class FloatMemoryMatrix {

    private final long rows;
    private final long columns;
    private final MemorySegment memorySegment;

    public FloatMemoryMatrix(long rows, long columns, MemorySegment memorySegment) {
        this.rows = rows;
        this.columns = columns;
        this.memorySegment = memorySegment;
    }

    public float getAt(long row, long column) {
        validateCoordinates(row, column);
        return memorySegment.getAtIndex(ValueLayout.JAVA_FLOAT, row * columns + column);
    }

    public void setAt(long row, long column, float value) {
        validateCoordinates(row, column);
        memorySegment.setAtIndex(ValueLayout.JAVA_FLOAT, row * columns + column, value);
    }

    public void setData(float[][] data) {
        int dataRows = data.length;
        int dataColumns = data[0].length;
        for (int i = 0; i < dataRows; i++) {
            float[] row = data[i];
            for (int j = 0; j < dataColumns; j++) {
                setAt(i, j, row[j]);
            }
        }
    }

    public long getByteSize() {
        return memorySegment.byteSize();
    }

    public MemorySegment getMemorySegment() {
        return memorySegment;
    }

    private void validateCoordinates(long row, long column) {
        if (row < 0L || column < 0L) {
            throw new IllegalArgumentException(
                    "Both row and column must not be negative, but got: (" + row + ", " + column + ")");
        }
        if (row >= rows || column >= columns) {
            throw new IllegalArgumentException(
                    "Both row and column must not exceed matrix size, but got: ("
                            + row + ", " + column + ") for matrix with size ("+ rows + ", " + columns + ")");
        }
    }
}
