package com.dmytrobilokha.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;

public class MemoryMatrixFactory {

    private final Arena arena;

    private MemoryMatrixFactory(Arena arena) {
        this.arena = arena;
    }

    public static MemoryMatrixFactory newInstance() {
        return new MemoryMatrixFactory(Arena.ofAuto());
    }

    public FloatMemoryMatrix createFloatMatrix(long rows, long columns) {
        return new FloatMemoryMatrix(rows, columns, arena.allocate(ValueLayout.JAVA_FLOAT, rows * columns));
    }

}
