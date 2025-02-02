package com.dmytrobilokha.verification;

import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *  This implementation was copy-pasted from org.apache.commons.math3.linear.BlockRealMatrix with some changes:
 *  - all methods not used have been removed;
 *  - only one primitive private constructor;
 *  - added factory methods to create a matrix;
 *  - removed not needed checks;
 *  - added some missing, but useful methods (element-by-element multiply, sum, etc.);
 *  - made internal blocks array read-only
 *  - data type changed to float
 */
public class FloatMatrix {

    public static final int BLOCK_SIZE = 52 * 2;
    private static final Random randomGenerator = new SecureRandom("Deterministic".getBytes(StandardCharsets.UTF_8));
    private final float[][] blocks;
    private final int rows;
    private final int columns;
    private final int blockRows;
    private final int blockColumns;

    private FloatMatrix(int rows, int columns, int blockRows, int blockColumns, float[][] blockData) {
        this.rows = rows;
        this.columns = columns;
        this.blockRows = blockRows;
        this.blockColumns = blockColumns;
        blocks = blockData;
    }

    public static FloatMatrix ofZeros(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new FloatMatrix(rows, columns, blockRows, blockColumns, createBlocksLayout(rows, columns));
    }

    public static FloatMatrix ofOnes(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new FloatMatrix(rows, columns, blockRows, blockColumns, createFilledBlocksLayout(rows, columns, () -> 1f));
    }

    public static FloatMatrix ofSndRandoms(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new FloatMatrix(
                rows, columns, blockRows, blockColumns, createFilledBlocksLayout(rows, columns, () -> (float) randomGenerator.nextGaussian()));
    }

    public static FloatMatrix ofUniRandoms(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new FloatMatrix(rows, columns, blockRows, blockColumns, createFilledBlocksLayout(rows, columns, randomGenerator::nextFloat));
    }

    public static FloatMatrix ofZerosSizedAs(FloatMatrix m) {
        return new FloatMatrix(m.rows, m.columns, m.blockRows, m.blockColumns, createBlocksLayout(m.rows, m.columns));
    }

    public static FloatMatrix ofOnesSizedAs(FloatMatrix m) {
        return new FloatMatrix(m.rows, m.columns, m.blockRows, m.blockColumns, createFilledBlocksLayout(m.rows, m.columns, () -> 1f));
    }

    public static FloatMatrix ofValuesSizedAs(float value, FloatMatrix m) {
        return new FloatMatrix(m.rows, m.columns, m.blockRows, m.blockColumns, createFilledBlocksLayout(m.rows, m.columns, () -> value));
    }

    public static FloatMatrix with2dArray(float[][] array) {
        int rows = array.length;
        if (rows < 1) {
            throw new IllegalArgumentException("2D array must have rows");
        }
        int columns = array[0].length;
        if (columns < 1) {
            throw new IllegalArgumentException("2D array must have columns");
        }
        for (float[] row : array) {
            if (row.length != columns) {
                throw new IllegalArgumentException("All rows should have the same number of columns");
            }
        }
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new FloatMatrix(rows, columns, blockRows, blockColumns, toBlocksLayout(array));
    }

    private static float[][] toBlocksLayout(final float[][] rawData) {
        final int rows = rawData.length;
        final int columns = rawData[0].length;
        final int blockRows = (rows    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // convert array
        final float[][] blocks = new float[blockRows * blockColumns][];
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            final int iHeight = pEnd - pStart;
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                final int jWidth = qEnd - qStart;

                // allocate new block
                final float[] block = new float[iHeight * jWidth];
                blocks[blockIndex] = block;

                // copy data
                int index = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    System.arraycopy(rawData[p], qStart, block, index, jWidth);
                    index += jWidth;
                }
                ++blockIndex;
            }
        }
        return blocks;
    }

    private static float[][] createBlocksLayout(final int rows, final int columns) {
        final int blockRows = (rows    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;

        final float[][] blocks = new float[blockRows * blockColumns][];
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            final int iHeight = pEnd - pStart;
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                final int jWidth = qEnd - qStart;
                blocks[blockIndex] = new float[iHeight * jWidth];
                ++blockIndex;
            }
        }
        return blocks;
    }

    private static float[][] createFilledBlocksLayout(final int rows, final int columns, FloatSupplier fillValueSupplier) {
        final int blockRows = (rows    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;

        final float[][] blocks = new float[blockRows * blockColumns][];
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            final int iHeight = pEnd - pStart;
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                final int jWidth = qEnd - qStart;
                float[] block = new float[iHeight * jWidth];
                for (int k = 0; k < block.length; k++) {
                    block[k] = fillValueSupplier.getAsFloat();
                }
                blocks[blockIndex] = block;
                ++blockIndex;
            }
        }
        return blocks;
    }


    public int getRowDimension() {
        return rows;
    }

    public int getColumnDimension() {
        return columns;
    }

    public FloatMatrix add(FloatMatrix m) {
        ensureSameSize(m);
        float[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final float[] outBlock = outBlocks[blockIndex];
            final float[] tBlock = blocks[blockIndex];
            final float[] mBlock = m.blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] + mBlock[k];
            }
        }
        return new FloatMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public FloatMatrix subtract(FloatMatrix m) {
        ensureSameSize(m);
        float[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final float[] outBlock = outBlocks[blockIndex];
            final float[] tBlock = blocks[blockIndex];
            final float[] mBlock = m.blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] - mBlock[k];
            }
        }
        return new FloatMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public FloatMatrix multiplyElements(FloatMatrix m) {
        ensureSameSize(m);
        float[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final float[] outBlock = outBlocks[blockIndex];
            final float[] tBlock = blocks[blockIndex];
            final float[] mBlock = m.blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] * mBlock[k];
            }
        }
        return new FloatMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    record Coordinates(int row, int column) {}

    private Coordinates calculateInBlocksCoordinates(int row, int column, int columns, int blockColumns) {
        final int iBlock = row / BLOCK_SIZE;
        final int jBlock = column / BLOCK_SIZE;
        final int k = (row - iBlock * BLOCK_SIZE) * calculateBlockWidth(jBlock, columns, blockColumns) +
                (column - jBlock * BLOCK_SIZE);
        return new Coordinates(iBlock * blockColumns + jBlock, k);
    }

    public FloatMatrix divideRows(FloatMatrix m) {
        if (this.rows != m.rows || m.columns != 1) {
            throw new IllegalArgumentException("("
                    + this.rows + ", " + this.columns + ") cannot have rows divided by (" + m.rows + ", " + m.columns + ")");
        }
        float[][] outBlocks = createBlocksLayout(rows, columns);
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                var divisorCoordinates = calculateInBlocksCoordinates(p, 0, 1, 1);
                float divisor = m.blocks[divisorCoordinates.row()][divisorCoordinates.column()];
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                    final float[] block = blocks[iBlock * blockColumns + jBlock];
                    final float[] outBlock = outBlocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        outBlock[k] = block[k] / divisor;
                        ++k;
                    }
                }
            }
        }
        return new FloatMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public FloatMatrix sumPerRow() {
        float[][] outBlocks = createBlocksLayout(rows, 1);
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                float rowSum = 0;
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                    final float[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        rowSum += block[k];
                        ++k;
                    }
                }
                var resultCoordinates = calculateInBlocksCoordinates(p, 0, 1, 1);
                outBlocks[resultCoordinates.row()][resultCoordinates.column()] = rowSum;
            }
        }
        return new FloatMatrix(rows, 1, blockRows, 1, outBlocks);
    }

    public String toString(String columnSeparator, String rowSeparator) {
        var outputBuilder = new StringBuilder();
        float[][] data = getData();
        for (int row = 0; row < rows; row++) {
            float[] rowData = data[row];
            if (row != 0) {
                outputBuilder.append(rowSeparator);
            }
            for (int column = 0; column < columns; column++) {
                if (column != 0) {
                    outputBuilder.append(columnSeparator);
                }
                outputBuilder.append(rowData[column]);
            }
        }
        return outputBuilder.toString();
    }

    public FloatMatrix scalarMultiply(float s) {
        float[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final float[] outBlock = outBlocks[blockIndex];
            final float[] tBlock = blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] * s;
            }
        }
        return new FloatMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public FloatMatrix scalarDivide(float s) {
        float[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final float[] outBlock = outBlocks[blockIndex];
            final float[] tBlock = blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] / s;
            }
        }
        return new FloatMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public FloatMatrix scalarAdd(float s) {
        float[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final float[] outBlock = outBlocks[blockIndex];
            final float[] tBlock = blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] + s;
            }
        }
        return new FloatMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public float sum() {
        float sum = 0;
        for (int blockIndex = 0; blockIndex < blocks.length; ++blockIndex) {
            final float[] tBlock = blocks[blockIndex];
            for (int k = 0; k < tBlock.length; ++k) {
                sum += tBlock[k];
            }
        }
        return sum;
    }

    public FloatMatrix prependColumn(float fillValue) {
        int outRows = rows;
        int outColumns = columns + 1;
        int outBlockRows = blockRows;
        int outBlockColumns = (outColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float[][] outBlocks = createBlocksLayout(outRows, outColumns);
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                var firstEntryInRowCoordinates = calculateInBlocksCoordinates(p, 0, outColumns, outBlockColumns);
                outBlocks[firstEntryInRowCoordinates.row()][firstEntryInRowCoordinates.column()] = fillValue;
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                    final float[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        var resultCoordinates = calculateInBlocksCoordinates(p, q + 1, outColumns, outBlockColumns);
                        outBlocks[resultCoordinates.row()][resultCoordinates.column()] = block[k];
                        ++k;
                    }
                }
            }
        }
        return new FloatMatrix(outRows, outColumns, outBlockRows, outBlockColumns, outBlocks);
    }

    public FloatMatrix multiply(FloatMatrix m) {
        if (columns != m.rows) {
            throw new IllegalArgumentException("This matrix has " + columns + " columns, other has " + m.rows + " rows");
        }
        int outRows = rows;
        int outColumns = m.columns;
        int outBlockRows = (outRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int outBlockColumns = (outColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float[][] outBlocks = createBlocksLayout(outRows, outColumns);
        // perform multiplication block-wise, to ensure good cache behavior
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < outBlockRows; ++iBlock) {

            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);

            for (int jBlock = 0; jBlock < outBlockColumns; ++jBlock) {
                final int jWidth = calculateBlockWidth(jBlock, outColumns, outBlockColumns);
                final int jWidth2 = jWidth  + jWidth;
                final int jWidth3 = jWidth2 + jWidth;
                final int jWidth4 = jWidth3 + jWidth;

                // select current block
                final float[] outBlock = outBlocks[blockIndex];

                // perform multiplication on current block
                for (int kBlock = 0; kBlock < blockColumns; ++kBlock) {
                    final int kWidth = blockWidth(kBlock);
                    final float[] tBlock = blocks[iBlock * blockColumns + kBlock];
                    final float[] mBlock = m.blocks[kBlock * m.blockColumns + jBlock];
                    int k = 0;
                    for (int p = pStart; p < pEnd; ++p) {
                        final int lStart = (p - pStart) * kWidth;
                        final int lEnd = lStart + kWidth;
                        for (int nStart = 0; nStart < jWidth; ++nStart) {
                            float sum = 0;
                            int l = lStart;
                            int n = nStart;
                            while (l < lEnd - 3) {
                                sum += tBlock[l] * mBlock[n] +
                                        tBlock[l + 1] * mBlock[n + jWidth] +
                                        tBlock[l + 2] * mBlock[n + jWidth2] +
                                        tBlock[l + 3] * mBlock[n + jWidth3];
                                l += 4;
                                n += jWidth4;
                            }
                            while (l < lEnd) {
                                sum += tBlock[l++] * mBlock[n];
                                n += jWidth;
                            }
                            outBlock[k] += sum;
                            ++k;
                        }
                    }
                }
                // go to next block
                ++blockIndex;
            }
        }
        return new FloatMatrix(outRows, outColumns, outBlockRows, outBlockColumns, outBlocks);
    }

    public FloatMatrix transpose() {
        int outRows = columns;
        int outColumns = rows;
        int outBlockRows = (outRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int outBlockColumns = (outColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float[][] outBlocks = createBlocksLayout(outRows, outColumns);
        // perform transpose block-wise, to ensure good cache behavior
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockColumns; ++iBlock) {
            for (int jBlock = 0; jBlock < blockRows; ++jBlock) {
                // transpose current block
                final float[] outBlock = outBlocks[blockIndex];
                final float[] tBlock = blocks[jBlock * blockColumns + iBlock];
                final int pStart = iBlock * BLOCK_SIZE;
                final int pEnd = Math.min(pStart + BLOCK_SIZE, columns);
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = Math.min(qStart + BLOCK_SIZE, rows);
                int k = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    final int lInc = pEnd - pStart;
                    int l = p - pStart;
                    for (int q = qStart; q < qEnd; ++q) {
                        outBlock[k] = tBlock[l];
                        ++k;
                        l+= lInc;
                    }
                }
                // go to next block
                ++blockIndex;
            }
        }
        return new FloatMatrix(outRows, outColumns, outBlockRows, outBlockColumns, outBlocks);
    }

    public FloatMatrix apply(FloatUnaryOperator operator) {
        float[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final float[] outBlock = outBlocks[blockIndex];
            final float[] tBlock = blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = operator.applyAsFloat(tBlock[k]);
            }
        }
        return new FloatMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public int[] indexOfHighestPerRow() {
        int[] result = new int[rows];
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                int indexOfHighest = 0;
                float highest = blocks[iBlock * blockColumns][(p - pStart) * blockWidth(0)];
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                    final float[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        float value = block[k];
                        if (value > highest) {
                            highest = value;
                            indexOfHighest = q;
                        }
                        ++k;
                    }
                }
                result[p] = indexOfHighest;
            }
        }
        return result;
    }

    public float[][] getData() {
        final float[][] data = new float[rows][columns];
        final int lastColumns = columns - (blockColumns - 1) * BLOCK_SIZE;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            int regularPos = 0;
            int lastPos = 0;
            for (int p = pStart; p < pEnd; ++p) {
                final float[] dataP = data[p];
                int blockIndex = iBlock * blockColumns;
                int dataPos = 0;
                for (int jBlock = 0; jBlock < blockColumns - 1; ++jBlock) {
                    System.arraycopy(blocks[blockIndex++], regularPos, dataP, dataPos, BLOCK_SIZE);
                    dataPos += BLOCK_SIZE;
                }
                System.arraycopy(blocks[blockIndex], lastPos, dataP, dataPos, lastColumns);
                regularPos += BLOCK_SIZE;
                lastPos    += lastColumns;
            }
        }
        return data;
    }

    public FloatMatrix cutOffFirstRows(int numberOfRows) {
        return getSubMatrix(numberOfRows, rows - 1, 0, columns - 1);
    }

    public List<FloatMatrix> splitRowsInBatches(int batchSize) {
       if (batchSize < 1) {
           throw new IllegalArgumentException("Minimum batch size is 1, but got " + batchSize);
       }
       var batches = new ArrayList<FloatMatrix>();
       int leftoverBatchSize = rows % batchSize;
       for (int startRow = 0, endRow = batchSize - 1; endRow < rows; startRow += batchSize, endRow += batchSize) {
           batches.add(getSubMatrix(startRow, endRow, 0, columns - 1));
       }
       if (leftoverBatchSize > 0) {
           batches.add(getSubMatrix(rows - leftoverBatchSize, rows - 1, 0, columns - 1));
       }
       return batches;
    }

    public FloatMatrix getSubMatrix(final int startRow, final int endRow,
                                    final int startColumn,
                                    final int endColumn) {
        // safety checks
        if (startRow < 0 || startColumn < 0) {
            throw new IllegalArgumentException("Both start row and start column can not be negative");
        }
        if (endRow >= rows || endColumn >= columns) {
            throw new IllegalArgumentException("Both end row and end column can not excess matrix size");
        }
        if (startRow >= endRow || startColumn >= endColumn) {
            throw new IllegalArgumentException("Start coordinates can not be higher or equal to end coordinates");
        }
        int outRows = endRow - startRow + 1;
        int outColumns = endColumn - startColumn + 1;
        int outBlockRows = (outRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int outBlockColumns = (outColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float[][] outBlocks = createBlocksLayout(outRows, outColumns);
        // compute blocks shifts
        final int blockStartRow = startRow / BLOCK_SIZE;
        final int rowsShift = startRow % BLOCK_SIZE;
        final int blockStartColumn = startColumn / BLOCK_SIZE;
        final int columnsShift = startColumn % BLOCK_SIZE;

        // perform extraction block-wise, to ensure good cache behavior
        int pBlock = blockStartRow;
        for (int iBlock = 0; iBlock < outBlockRows; ++iBlock) {
            final int iHeight = calculateBlockHeight(iBlock, outRows, outBlockRows);
            int qBlock = blockStartColumn;
            for (int jBlock = 0; jBlock < outBlockColumns; ++jBlock) {
                final int jWidth = calculateBlockWidth(jBlock, outColumns, outBlockColumns);

                // handle one block of the output matrix
                final int outIndex = iBlock * outBlockColumns + jBlock;
                final float[] outBlock = outBlocks[outIndex];
                final int index = pBlock * blockColumns + qBlock;
                final int width = blockWidth(qBlock);

                final int heightExcess = iHeight + rowsShift - BLOCK_SIZE;
                final int widthExcess = jWidth + columnsShift - BLOCK_SIZE;
                if (heightExcess > 0) {
                    // the submatrix block spans on two blocks rows from the original matrix
                    if (widthExcess > 0) {
                        // the submatrix block spans on two blocks columns from the original matrix
                        final int width2 = blockWidth(qBlock + 1);
                        copyBlockPart(blocks[index], width,
                                rowsShift, BLOCK_SIZE,
                                columnsShift, BLOCK_SIZE,
                                outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + 1], width2,
                                rowsShift, BLOCK_SIZE,
                                0, widthExcess,
                                outBlock, jWidth, 0, jWidth - widthExcess);
                        copyBlockPart(blocks[index + blockColumns], width,
                                0, heightExcess,
                                columnsShift, BLOCK_SIZE,
                                outBlock, jWidth, iHeight - heightExcess, 0);
                        copyBlockPart(blocks[index + blockColumns + 1], width2,
                                0, heightExcess,
                                0, widthExcess,
                                outBlock, jWidth, iHeight - heightExcess, jWidth - widthExcess);
                    } else {
                        // the submatrix block spans on one block column from the original matrix
                        copyBlockPart(blocks[index], width,
                                rowsShift, BLOCK_SIZE,
                                columnsShift, jWidth + columnsShift,
                                outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + blockColumns], width,
                                0, heightExcess,
                                columnsShift, jWidth + columnsShift,
                                outBlock, jWidth, iHeight - heightExcess, 0);
                    }
                } else {
                    // the submatrix block spans on one block row from the original matrix
                    if (widthExcess > 0) {
                        // the submatrix block spans on two blocks columns from the original matrix
                        final int width2 = blockWidth(qBlock + 1);
                        copyBlockPart(blocks[index], width,
                                rowsShift, iHeight + rowsShift,
                                columnsShift, BLOCK_SIZE,
                                outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + 1], width2,
                                rowsShift, iHeight + rowsShift,
                                0, widthExcess,
                                outBlock, jWidth, 0, jWidth - widthExcess);
                    } else {
                        // the submatrix block spans on one block column from the original matrix
                        copyBlockPart(blocks[index], width,
                                rowsShift, iHeight + rowsShift,
                                columnsShift, jWidth + columnsShift,
                                outBlock, jWidth, 0, 0);
                    }
                }
                ++qBlock;
            }
            ++pBlock;
        }
        return new FloatMatrix(outRows, outColumns, outBlockRows, outBlockColumns, outBlocks);
    }

    private void copyBlockPart(final float[] srcBlock, final int srcWidth,
                               final int srcStartRow, final int srcEndRow,
                               final int srcStartColumn, final int srcEndColumn,
                               final float[] dstBlock, final int dstWidth,
                               final int dstStartRow, final int dstStartColumn) {
        final int length = srcEndColumn - srcStartColumn;
        int srcPos = srcStartRow * srcWidth + srcStartColumn;
        int dstPos = dstStartRow * dstWidth + dstStartColumn;
        for (int srcRow = srcStartRow; srcRow < srcEndRow; ++srcRow) {
            System.arraycopy(srcBlock, srcPos, dstBlock, dstPos, length);
            srcPos += srcWidth;
            dstPos += dstWidth;
        }
    }

    private void ensureSameSize(FloatMatrix other) {
        if (this.rows != other.rows || this.columns != other.columns) {
            throw new IllegalArgumentException("Matrix sizes are not the same: ("
            + this.rows + ", " + this.columns + ") vs (" + other.rows + ", " + other.columns+ ")");
        }
    }

    private static void ensureCreatableSize(int rows, int columns) {
        if (rows < 1 || columns < 1) {
            throw new IllegalArgumentException("(" + rows + ", " + columns + ") is not a valid size");
        }
    }

    private int blockHeight(final int blockRow) {
        return (blockRow == blockRows - 1) ? rows - blockRow * BLOCK_SIZE : BLOCK_SIZE;
    }

    private int blockWidth(final int blockColumn) {
        return (blockColumn == blockColumns - 1) ? columns - blockColumn * BLOCK_SIZE : BLOCK_SIZE;
    }

    private int calculateBlockWidth(int blockColumn, int columns, int blockColumns) {
        return (blockColumn == blockColumns - 1) ? columns - blockColumn * BLOCK_SIZE : BLOCK_SIZE;
    }

    private int calculateBlockHeight(int blockRow, int rows, int blockRows) {
        return (blockRow == blockRows - 1) ? rows - blockRow * BLOCK_SIZE : BLOCK_SIZE;
    }

    @FunctionalInterface
    public interface FloatSupplier {
       float getAsFloat();
    }

    @FunctionalInterface
    public interface FloatUnaryOperator {
        float applyAsFloat(float input);
    }

}
