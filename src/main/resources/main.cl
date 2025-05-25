__kernel void addMatrices(__global const float* a, __global const float* b, __global float* result, const unsigned long elementCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    for (unsigned long i = id; i < elementCount; i += globalSize) {
        result[i] = a[i] + b[i];
    }
}

__kernel void addMatrices16(__global const float16* a, __global const float16* b, __global float16* result, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        result[i] = a[i] + b[i];
    }
}

__kernel void addMatrices8(__global const float8* a, __global const float8* b, __global float8* result, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        result[i] = a[i] + b[i];
    }
}

__kernel void addMatrices4(__global const float4* a, __global const float4* b, __global float4* result, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        result[i] = a[i] + b[i];
    }
}

__kernel void addMatrices2(__global const float2* a, __global const float2* b, __global float2* result, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        result[i] = a[i] + b[i];
    }
}

__kernel void addMatricesLeftover(__global const float* a, __global const float* b, __global float* result, const unsigned long startIndex, const unsigned long endIndex) {
    for (unsigned long i = startIndex; i < endIndex; i++) {
        result[i] = a[i] + b[i];
    }
}

__kernel void sigmoidElements(__global const float* input, __global float* output, const unsigned long elementCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    for (unsigned long i = id; i < elementCount; i += globalSize) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}

__kernel void sigmoidElements16(__global const float16* input, __global float16* output, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    const float16 vectorOfOnes = (float16)(1.0f);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        output[i] = vectorOfOnes / (vectorOfOnes + exp(-input[i]));
    }
}

__kernel void sigmoidElements8(__global const float8* input, __global float8* output, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    const float8 vectorOfOnes = (float8)(1.0f);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        output[i] = vectorOfOnes / (vectorOfOnes + exp(-input[i]));
    }
}

__kernel void sigmoidElements4(__global const float4* input, __global float4* output, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    const float4 vectorOfOnes = (float4)(1.0f);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        output[i] = vectorOfOnes / (vectorOfOnes + exp(-input[i]));
    }
}

__kernel void sigmoidElements2(__global const float2* input, __global float2* output, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    const float2 vectorOfOnes = (float2)(1.0f);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        output[i] = vectorOfOnes / (vectorOfOnes + exp(-input[i]));
    }
}

__kernel void sigmoidElementsLeftover(__global const float* input, __global float* output, const unsigned long startIndex, const unsigned long endIndex) {
    for (unsigned long i = startIndex; i < endIndex; i++) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}
__kernel void sigmoidElementsN(__global const float* input, __global float* output, const unsigned long elementCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    for (unsigned long i = id; i < elementCount; i += globalSize) {
        output[i] = 1.0f / (1.0f + native_exp(-input[i]));
    }
}

__kernel void sigmoidElements16N(__global const float16* input, __global float16* output, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    const float16 vectorOfOnes = (float16)(1.0f);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        output[i] = vectorOfOnes / (vectorOfOnes + native_exp(-input[i]));
    }
}

__kernel void sigmoidElements8N(__global const float8* input, __global float8* output, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    const float8 vectorOfOnes = (float8)(1.0f);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        output[i] = vectorOfOnes / (vectorOfOnes + native_exp(-input[i]));
    }
}

__kernel void sigmoidElements4N(__global const float4* input, __global float4* output, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    const float4 vectorOfOnes = (float4)(1.0f);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        output[i] = vectorOfOnes / (vectorOfOnes + native_exp(-input[i]));
    }
}

__kernel void sigmoidElements2N(__global const float2* input, __global float2* output, const unsigned long vectorCount) {
    const unsigned long id = get_global_id(0);
    const unsigned long globalSize = get_global_size(0);
    const float2 vectorOfOnes = (float2)(1.0f);
    for (unsigned long i = id; i < vectorCount; i += globalSize) {
        output[i] = vectorOfOnes / (vectorOfOnes + native_exp(-input[i]));
    }
}

__kernel void sigmoidElementsLeftoverN(__global const float* input, __global float* output, const unsigned long startIndex, const unsigned long endIndex) {
    for (unsigned long i = startIndex; i < endIndex; i++) {
        output[i] = 1.0f / (1.0f + native_exp(-input[i]));
    }
}

__kernel void multiplyMatricesSimple(
        __global const float* a,
        __global const float* b,
        __global float* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
        ) {
    const unsigned long globalRow = get_global_id(1);
    const unsigned long globalColumn = get_global_id(0);
    float resultElement = 0.0f;
    for (unsigned long k = 0; k < kDimension; k++) {
        resultElement += a[globalRow * kDimension + k] * b[k * nDimension + globalColumn];
    }
    result[globalRow * nDimension + globalColumn] = resultElement;
}

__kernel void padRightBottom(
        __global const float* input,
        __global float* output,
        const unsigned long inputRows,
        const unsigned long inputColumns,
        const unsigned long outputRows,
        const unsigned long outputColumns
) {
    const unsigned int row = get_global_id(1);
    const unsigned int column = get_global_id(0);
    if (row >= outputRows || column >= outputColumns) {
        return;
    }
    float value;
    if (row < inputRows && column < inputColumns) {
        value = input[row * inputColumns + column];
    } else {
        value = 0.0f;
    }
    output[row * outputColumns + column] = value;
}

__kernel void unpadRightBottom(
        __global const float* input,
        __global float* output,
        const unsigned long inputColumns,
        const unsigned long outputColumns
) {
    const unsigned int row = get_global_id(1);
    const unsigned int column = get_global_id(0);
    output[row * outputColumns + column] = input[row * inputColumns + column];
}

__kernel void multiplyMatricesTile32(
        __global const float* a,
        __global const float* b,
        __global float* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int tileSize = 32;
    const unsigned int localRow = get_local_id(1);
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + localRow;
    const unsigned int globalColumn = tileSize * get_group_id(0) + localColumn;
    __local float submatrixA[tileSize][tileSize];
    __local float submatrixB[tileSize][tileSize];
    float resultElement = 0.0f;
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = tileSize * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int k = 0; k < tileSize; k++) {
            resultElement += submatrixA[localRow][k] * submatrixB[k][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension + globalColumn] = resultElement;
}

__kernel void multiplyMatricesTile16(
        __global const float* a,
        __global const float* b,
        __global float* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int tileSize = 16;
    const unsigned int localRow = get_local_id(1);
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + localRow;
    const unsigned int globalColumn = tileSize * get_group_id(0) + localColumn;
    __local float submatrixA[tileSize][tileSize];
    __local float submatrixB[tileSize][tileSize];
    float resultElement = 0.0f;
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = tileSize * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int k = 0; k < tileSize; k++) {
            resultElement += submatrixA[localRow][k] * submatrixB[k][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension + globalColumn] = resultElement;
}

__kernel void multiplyMatricesTile32W8(
        __global const float* a,
        __global const float* b,
        __global float* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workPerThread = 8;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / workPerThread;
    const unsigned int localRow = get_local_id(1);
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + localRow;
    const unsigned int globalColumn = tileSize * get_group_id(0) + localColumn;
    __local float submatrixA[tileSize][tileSize];
    __local float submatrixB[tileSize][tileSize];
    float resultElements[workPerThread];
    for (unsigned int i = 0; i < workPerThread; i++) {
        resultElements[i] = 0.0f;
    }
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = tileSize * tile + localColumn;
        for (unsigned int i = 0; i < workPerThread; i++) {
            submatrixA[localRow][localColumn + i * maxLocalColumn] = a[globalRow * kDimension + tiledColumn + i * maxLocalColumn];
            submatrixB[localRow][localColumn + i * maxLocalColumn] = b[tiledRow * nDimension + globalColumn + i * maxLocalColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int k = 0; k < tileSize; k++) {
            const float submatrixElement = submatrixA[localRow][k];
            for (unsigned int i = 0; i < workPerThread; i++) {
                resultElements[i] += submatrixElement * submatrixB[k][localColumn + i * maxLocalColumn];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (unsigned int i = 0; i < workPerThread; i++) {
        result[globalRow * nDimension + globalColumn + i * maxLocalColumn] = resultElements[i];
    }
}

__kernel void multiplyMatricesTile32W4(
        __global const float* a,
        __global const float* b,
        __global float* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workPerThread = 4;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / workPerThread;
    const unsigned int localRow = get_local_id(1);
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + localRow;
    const unsigned int globalColumn = tileSize * get_group_id(0) + localColumn;
    __local float submatrixA[tileSize][tileSize];
    __local float submatrixB[tileSize][tileSize];
    float resultElements[workPerThread];
    for (unsigned int i = 0; i < workPerThread; i++) {
        resultElements[i] = 0.0f;
    }
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = tileSize * tile + localColumn;
        for (unsigned int i = 0; i < workPerThread; i++) {
            submatrixA[localRow][localColumn + i * maxLocalColumn] = a[globalRow * kDimension + tiledColumn + i * maxLocalColumn];
            submatrixB[localRow][localColumn + i * maxLocalColumn] = b[tiledRow * nDimension + globalColumn + i * maxLocalColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int k = 0; k < tileSize; k++) {
            const float submatrixElement = submatrixA[localRow][k];
            for (unsigned int i = 0; i < workPerThread; i++) {
                resultElements[i] += submatrixElement * submatrixB[k][localColumn + i * maxLocalColumn];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (unsigned int i = 0; i < workPerThread; i++) {
        result[globalRow * nDimension + globalColumn + i * maxLocalColumn] = resultElements[i];
    }
}

__kernel void multiplyMatricesTile16W4(
        __global const float* a,
        __global const float* b,
        __global float* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workPerThread = 4;
    const unsigned int tileSize = 16;
    const unsigned int maxLocalColumn = tileSize / workPerThread;
    const unsigned int localRow = get_local_id(1);
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + localRow;
    const unsigned int globalColumn = tileSize * get_group_id(0) + localColumn;
    __local float submatrixA[tileSize][tileSize];
    __local float submatrixB[tileSize][tileSize];
    float resultElements[workPerThread];
    for (unsigned int i = 0; i < workPerThread; i++) {
        resultElements[i] = 0.0f;
    }
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = tileSize * tile + localColumn;
        for (unsigned int i = 0; i < workPerThread; i++) {
            submatrixA[localRow][localColumn + i * maxLocalColumn] = a[globalRow * kDimension + tiledColumn + i * maxLocalColumn];
            submatrixB[localRow][localColumn + i * maxLocalColumn] = b[tiledRow * nDimension + globalColumn + i * maxLocalColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int k = 0; k < tileSize; k++) {
            const float submatrixElement = submatrixA[localRow][k];
            for (unsigned int i = 0; i < workPerThread; i++) {
                resultElements[i] += submatrixElement * submatrixB[k][localColumn + i * maxLocalColumn];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (unsigned int i = 0; i < workPerThread; i++) {
        result[globalRow * nDimension + globalColumn + i * maxLocalColumn] = resultElements[i];
    }
}

__kernel void multiplyMatricesTile32V4(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localRow = get_local_id(1);
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + localRow;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[tileSize][maxLocalColumn];
    __local float4 submatrixB[tileSize][maxLocalColumn];
    float4 resultElements = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA = submatrixA[localRow][v];
            resultElements += vectorA.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements += vectorA.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements += vectorA.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements += vectorA.s3 * submatrixB[vectorWidth * v + 3][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements;
}

__kernel void multiplyMatricesTile32V8(
        __global const float8* a,
        __global const float8* b,
        __global float8* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int vectorWidth = 8;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localRow = get_local_id(1);
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + localRow;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float8 submatrixA[tileSize][maxLocalColumn];
    __local float8 submatrixB[tileSize][maxLocalColumn];
    float8 resultElements = (float8) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float8 vectorA = submatrixA[localRow][v];
            resultElements += vectorA.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements += vectorA.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements += vectorA.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements += vectorA.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements += vectorA.s4 * submatrixB[vectorWidth * v + 4][localColumn];
            resultElements += vectorA.s5 * submatrixB[vectorWidth * v + 5][localColumn];
            resultElements += vectorA.s6 * submatrixB[vectorWidth * v + 6][localColumn];
            resultElements += vectorA.s7 * submatrixB[vectorWidth * v + 7][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements;
}

__kernel void multiplyMatricesTile32V8H2(
        __global const float8* a,
        __global const float8* b,
        __global float8* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 2;
    const unsigned int vectorWidth = 8;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float8 submatrixA[tileSize][maxLocalColumn];
    __local float8 submatrixB[tileSize][maxLocalColumn];
    float8 resultElements0 = (float8) (0.0f);
    float8 resultElements1 = (float8) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float8 vectorA0 = submatrixA[localRow][v];
            const float8 vectorA1 = submatrixA[localRow + 1][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements0 += vectorA0.s4 * submatrixB[vectorWidth * v + 4][localColumn];
            resultElements1 += vectorA1.s4 * submatrixB[vectorWidth * v + 4][localColumn];
            resultElements0 += vectorA0.s5 * submatrixB[vectorWidth * v + 5][localColumn];
            resultElements1 += vectorA1.s5 * submatrixB[vectorWidth * v + 5][localColumn];
            resultElements0 += vectorA0.s6 * submatrixB[vectorWidth * v + 6][localColumn];
            resultElements1 += vectorA1.s6 * submatrixB[vectorWidth * v + 6][localColumn];
            resultElements0 += vectorA0.s7 * submatrixB[vectorWidth * v + 7][localColumn];
            resultElements1 += vectorA1.s7 * submatrixB[vectorWidth * v + 7][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
}

__kernel void multiplyMatricesTile32V8H8P(
        __global const float8* a,
        __global const float8* b,
        __global float8* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 8;
    const unsigned int vectorWidth = 8;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float8 submatrixA[2][tileSize * maxLocalColumn];
    __local float8 submatrixB[2][tileSize * maxLocalColumn];
    float8 resultElements0 = (float8) (0.0f);
    float8 resultElements1 = (float8) (0.0f);
    float8 resultElements2 = (float8) (0.0f);
    float8 resultElements3 = (float8) (0.0f);
    float8 resultElements4 = (float8) (0.0f);
    float8 resultElements5 = (float8) (0.0f);
    float8 resultElements6 = (float8) (0.0f);
    float8 resultElements7 = (float8) (0.0f);
    submatrixA[0][(localRow + 0) * maxLocalColumn + localColumn] = a[(globalRow + 0) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 1) * maxLocalColumn + localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 2) * maxLocalColumn + localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 3) * maxLocalColumn + localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 4) * maxLocalColumn + localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 5) * maxLocalColumn + localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 6) * maxLocalColumn + localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 7) * maxLocalColumn + localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + localColumn];
    submatrixB[0][(localRow + 0) * maxLocalColumn + localColumn] = b[(localRow + 0) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 1) * maxLocalColumn + localColumn] = b[(localRow + 1) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 2) * maxLocalColumn + localColumn] = b[(localRow + 2) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 3) * maxLocalColumn + localColumn] = b[(localRow + 3) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 4) * maxLocalColumn + localColumn] = b[(localRow + 4) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 5) * maxLocalColumn + localColumn] = b[(localRow + 5) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 6) * maxLocalColumn + localColumn] = b[(localRow + 6) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 7) * maxLocalColumn + localColumn] = b[(localRow + 7) * nDimension / vectorWidth + globalColumn];
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int nextTile = tile + 1;
        if (nextTile < numberOfTiles) {
            const unsigned int tiledRow = tileSize * nextTile + localRow;
            const unsigned int tiledColumn = maxLocalColumn * nextTile + localColumn;
            submatrixA[nextTile % 2][(localRow + 0) * maxLocalColumn + localColumn] = a[(globalRow + 0) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 1) * maxLocalColumn + localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 2) * maxLocalColumn + localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 3) * maxLocalColumn + localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 4) * maxLocalColumn + localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 5) * maxLocalColumn + localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 6) * maxLocalColumn + localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 7) * maxLocalColumn + localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + tiledColumn];
            submatrixB[nextTile % 2][(localRow + 0) * maxLocalColumn + localColumn] = b[(tiledRow + 0) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 1) * maxLocalColumn + localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 2) * maxLocalColumn + localColumn] = b[(tiledRow + 2) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 3) * maxLocalColumn + localColumn] = b[(tiledRow + 3) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 4) * maxLocalColumn + localColumn] = b[(tiledRow + 4) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 5) * maxLocalColumn + localColumn] = b[(tiledRow + 5) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 6) * maxLocalColumn + localColumn] = b[(tiledRow + 6) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 7) * maxLocalColumn + localColumn] = b[(tiledRow + 7) * nDimension / vectorWidth + globalColumn];
        }
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float8 vectorA0 = submatrixA[tile % 2][(localRow + 0) * maxLocalColumn + v];
            const float8 vectorA1 = submatrixA[tile % 2][(localRow + 1) * maxLocalColumn + v];
            const float8 vectorA2 = submatrixA[tile % 2][(localRow + 2) * maxLocalColumn + v];
            const float8 vectorA3 = submatrixA[tile % 2][(localRow + 3) * maxLocalColumn + v];
            const float8 vectorA4 = submatrixA[tile % 2][(localRow + 4) * maxLocalColumn + v];
            const float8 vectorA5 = submatrixA[tile % 2][(localRow + 5) * maxLocalColumn + v];
            const float8 vectorA6 = submatrixA[tile % 2][(localRow + 6) * maxLocalColumn + v];
            const float8 vectorA7 = submatrixA[tile % 2][(localRow + 7) * maxLocalColumn + v];
            resultElements0 += vectorA0.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s4 * submatrixB[tile % 2][(vectorWidth * v + 4) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s4 * submatrixB[tile % 2][(vectorWidth * v + 4) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s4 * submatrixB[tile % 2][(vectorWidth * v + 4) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s4 * submatrixB[tile % 2][(vectorWidth * v + 4) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s4 * submatrixB[tile % 2][(vectorWidth * v + 4) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s4 * submatrixB[tile % 2][(vectorWidth * v + 4) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s4 * submatrixB[tile % 2][(vectorWidth * v + 4) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s4 * submatrixB[tile % 2][(vectorWidth * v + 4) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s5 * submatrixB[tile % 2][(vectorWidth * v + 5) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s5 * submatrixB[tile % 2][(vectorWidth * v + 5) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s5 * submatrixB[tile % 2][(vectorWidth * v + 5) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s5 * submatrixB[tile % 2][(vectorWidth * v + 5) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s5 * submatrixB[tile % 2][(vectorWidth * v + 5) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s5 * submatrixB[tile % 2][(vectorWidth * v + 5) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s5 * submatrixB[tile % 2][(vectorWidth * v + 5) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s5 * submatrixB[tile % 2][(vectorWidth * v + 5) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s6 * submatrixB[tile % 2][(vectorWidth * v + 6) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s6 * submatrixB[tile % 2][(vectorWidth * v + 6) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s6 * submatrixB[tile % 2][(vectorWidth * v + 6) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s6 * submatrixB[tile % 2][(vectorWidth * v + 6) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s6 * submatrixB[tile % 2][(vectorWidth * v + 6) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s6 * submatrixB[tile % 2][(vectorWidth * v + 6) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s6 * submatrixB[tile % 2][(vectorWidth * v + 6) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s6 * submatrixB[tile % 2][(vectorWidth * v + 6) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s7 * submatrixB[tile % 2][(vectorWidth * v + 7) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s7 * submatrixB[tile % 2][(vectorWidth * v + 7) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s7 * submatrixB[tile % 2][(vectorWidth * v + 7) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s7 * submatrixB[tile % 2][(vectorWidth * v + 7) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s7 * submatrixB[tile % 2][(vectorWidth * v + 7) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s7 * submatrixB[tile % 2][(vectorWidth * v + 7) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s7 * submatrixB[tile % 2][(vectorWidth * v + 7) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s7 * submatrixB[tile % 2][(vectorWidth * v + 7) * maxLocalColumn + localColumn];
        }
    }
    result[(globalRow + 0) * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
    result[(globalRow + 2) * nDimension / vectorWidth + globalColumn] = resultElements2;
    result[(globalRow + 3) * nDimension / vectorWidth + globalColumn] = resultElements3;
    result[(globalRow + 4) * nDimension / vectorWidth + globalColumn] = resultElements4;
    result[(globalRow + 5) * nDimension / vectorWidth + globalColumn] = resultElements5;
    result[(globalRow + 6) * nDimension / vectorWidth + globalColumn] = resultElements6;
    result[(globalRow + 7) * nDimension / vectorWidth + globalColumn] = resultElements7;
}

__kernel void multiplyMatricesTile32V4H2(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 2;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[tileSize][maxLocalColumn];
    __local float4 submatrixB[tileSize][maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[localRow][v];
            const float4 vectorA1 = submatrixA[localRow + 1][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
}

__kernel void multiplyMatricesTile32V4H2P(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 2;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[2][tileSize * maxLocalColumn];
    __local float4 submatrixB[2][tileSize * maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    submatrixA[0][localRow * maxLocalColumn + localColumn] = a[globalRow * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 1) * maxLocalColumn + localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + localColumn];
    submatrixB[0][localRow * maxLocalColumn + localColumn] = b[localRow * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 1) * maxLocalColumn + localColumn] = b[(localRow + 1) * nDimension / vectorWidth + globalColumn];
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int nextTile = tile + 1;
        if (nextTile < numberOfTiles) {
            const unsigned int tiledRow = tileSize * nextTile + localRow;
            const unsigned int tiledColumn = maxLocalColumn * nextTile + localColumn;
            submatrixA[nextTile % 2][localRow * maxLocalColumn + localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 1) * maxLocalColumn + localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
            submatrixB[nextTile % 2][localRow * maxLocalColumn + localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 1) * maxLocalColumn + localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        }
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[tile % 2][localRow * maxLocalColumn + v];
            const float4 vectorA1 = submatrixA[tile % 2][(localRow + 1) * maxLocalColumn + v];
            resultElements0 += vectorA0.s0 * submatrixB[tile % 2][vectorWidth * v * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[tile % 2][vectorWidth * v * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
        }
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
}

__kernel void multiplyMatricesTile32V4H4(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 4;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[tileSize][maxLocalColumn];
    __local float4 submatrixB[tileSize][maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    float4 resultElements2 = (float4) (0.0f);
    float4 resultElements3 = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 2][localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 3][localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 2][localColumn] = b[(tiledRow + 2) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 3][localColumn] = b[(tiledRow + 3) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[localRow][v];
            const float4 vectorA1 = submatrixA[localRow + 1][v];
            const float4 vectorA2 = submatrixA[localRow + 2][v];
            const float4 vectorA3 = submatrixA[localRow + 3][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements2 += vectorA2.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements3 += vectorA3.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements2 += vectorA2.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements3 += vectorA3.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements2 += vectorA2.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements3 += vectorA3.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements2 += vectorA2.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements3 += vectorA3.s3 * submatrixB[vectorWidth * v + 3][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
    result[(globalRow + 2) * nDimension / vectorWidth + globalColumn] = resultElements2;
    result[(globalRow + 3) * nDimension / vectorWidth + globalColumn] = resultElements3;
}

__kernel void multiplyMatricesTile32V4H8(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 8;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[tileSize][maxLocalColumn];
    __local float4 submatrixB[tileSize][maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    float4 resultElements2 = (float4) (0.0f);
    float4 resultElements3 = (float4) (0.0f);
    float4 resultElements4 = (float4) (0.0f);
    float4 resultElements5 = (float4) (0.0f);
    float4 resultElements6 = (float4) (0.0f);
    float4 resultElements7 = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 2][localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 3][localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 4][localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 5][localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 6][localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 7][localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 2][localColumn] = b[(tiledRow + 2) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 3][localColumn] = b[(tiledRow + 3) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 4][localColumn] = b[(tiledRow + 4) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 5][localColumn] = b[(tiledRow + 5) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 6][localColumn] = b[(tiledRow + 6) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 7][localColumn] = b[(tiledRow + 7) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[localRow][v];
            const float4 vectorA1 = submatrixA[localRow + 1][v];
            const float4 vectorA2 = submatrixA[localRow + 2][v];
            const float4 vectorA3 = submatrixA[localRow + 3][v];
            const float4 vectorA4 = submatrixA[localRow + 4][v];
            const float4 vectorA5 = submatrixA[localRow + 5][v];
            const float4 vectorA6 = submatrixA[localRow + 6][v];
            const float4 vectorA7 = submatrixA[localRow + 7][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements2 += vectorA2.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements3 += vectorA3.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements4 += vectorA4.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements5 += vectorA5.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements6 += vectorA6.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements7 += vectorA7.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements2 += vectorA2.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements3 += vectorA3.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements4 += vectorA4.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements5 += vectorA5.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements6 += vectorA6.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements7 += vectorA7.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements2 += vectorA2.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements3 += vectorA3.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements4 += vectorA4.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements5 += vectorA5.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements6 += vectorA6.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements7 += vectorA7.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements2 += vectorA2.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements3 += vectorA3.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements4 += vectorA4.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements5 += vectorA5.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements6 += vectorA6.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements7 += vectorA7.s3 * submatrixB[vectorWidth * v + 3][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
    result[(globalRow + 2) * nDimension / vectorWidth + globalColumn] = resultElements2;
    result[(globalRow + 3) * nDimension / vectorWidth + globalColumn] = resultElements3;
    result[(globalRow + 4) * nDimension / vectorWidth + globalColumn] = resultElements4;
    result[(globalRow + 5) * nDimension / vectorWidth + globalColumn] = resultElements5;
    result[(globalRow + 6) * nDimension / vectorWidth + globalColumn] = resultElements6;
    result[(globalRow + 7) * nDimension / vectorWidth + globalColumn] = resultElements7;
}

__kernel void multiplyMatricesTile32V4H8P(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 8;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 32;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[2][tileSize * maxLocalColumn];
    __local float4 submatrixB[2][tileSize * maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    float4 resultElements2 = (float4) (0.0f);
    float4 resultElements3 = (float4) (0.0f);
    float4 resultElements4 = (float4) (0.0f);
    float4 resultElements5 = (float4) (0.0f);
    float4 resultElements6 = (float4) (0.0f);
    float4 resultElements7 = (float4) (0.0f);
    submatrixA[0][(localRow + 0) * maxLocalColumn + localColumn] = a[(globalRow + 0) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 1) * maxLocalColumn + localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 2) * maxLocalColumn + localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 3) * maxLocalColumn + localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 4) * maxLocalColumn + localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 5) * maxLocalColumn + localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 6) * maxLocalColumn + localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 7) * maxLocalColumn + localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + localColumn];
    submatrixB[0][(localRow + 0) * maxLocalColumn + localColumn] = b[(localRow + 0) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 1) * maxLocalColumn + localColumn] = b[(localRow + 1) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 2) * maxLocalColumn + localColumn] = b[(localRow + 2) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 3) * maxLocalColumn + localColumn] = b[(localRow + 3) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 4) * maxLocalColumn + localColumn] = b[(localRow + 4) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 5) * maxLocalColumn + localColumn] = b[(localRow + 5) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 6) * maxLocalColumn + localColumn] = b[(localRow + 6) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 7) * maxLocalColumn + localColumn] = b[(localRow + 7) * nDimension / vectorWidth + globalColumn];
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int nextTile = tile + 1;
        if (nextTile < numberOfTiles) {
            const unsigned int tiledRow = tileSize * nextTile + localRow;
            const unsigned int tiledColumn = maxLocalColumn * nextTile + localColumn;
            submatrixA[nextTile % 2][(localRow + 0) * maxLocalColumn + localColumn] = a[(globalRow + 0) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 1) * maxLocalColumn + localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 2) * maxLocalColumn + localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 3) * maxLocalColumn + localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 4) * maxLocalColumn + localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 5) * maxLocalColumn + localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 6) * maxLocalColumn + localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 7) * maxLocalColumn + localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + tiledColumn];
            submatrixB[nextTile % 2][(localRow + 0) * maxLocalColumn + localColumn] = b[(tiledRow + 0) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 1) * maxLocalColumn + localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 2) * maxLocalColumn + localColumn] = b[(tiledRow + 2) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 3) * maxLocalColumn + localColumn] = b[(tiledRow + 3) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 4) * maxLocalColumn + localColumn] = b[(tiledRow + 4) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 5) * maxLocalColumn + localColumn] = b[(tiledRow + 5) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 6) * maxLocalColumn + localColumn] = b[(tiledRow + 6) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 7) * maxLocalColumn + localColumn] = b[(tiledRow + 7) * nDimension / vectorWidth + globalColumn];
        }
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[tile % 2][(localRow + 0) * maxLocalColumn + v];
            const float4 vectorA1 = submatrixA[tile % 2][(localRow + 1) * maxLocalColumn + v];
            const float4 vectorA2 = submatrixA[tile % 2][(localRow + 2) * maxLocalColumn + v];
            const float4 vectorA3 = submatrixA[tile % 2][(localRow + 3) * maxLocalColumn + v];
            const float4 vectorA4 = submatrixA[tile % 2][(localRow + 4) * maxLocalColumn + v];
            const float4 vectorA5 = submatrixA[tile % 2][(localRow + 5) * maxLocalColumn + v];
            const float4 vectorA6 = submatrixA[tile % 2][(localRow + 6) * maxLocalColumn + v];
            const float4 vectorA7 = submatrixA[tile % 2][(localRow + 7) * maxLocalColumn + v];
            resultElements0 += vectorA0.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
        }
    }
    result[(globalRow + 0) * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
    result[(globalRow + 2) * nDimension / vectorWidth + globalColumn] = resultElements2;
    result[(globalRow + 3) * nDimension / vectorWidth + globalColumn] = resultElements3;
    result[(globalRow + 4) * nDimension / vectorWidth + globalColumn] = resultElements4;
    result[(globalRow + 5) * nDimension / vectorWidth + globalColumn] = resultElements5;
    result[(globalRow + 6) * nDimension / vectorWidth + globalColumn] = resultElements6;
    result[(globalRow + 7) * nDimension / vectorWidth + globalColumn] = resultElements7;
}

__kernel void multiplyMatricesTile64V4H2(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 2;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 64;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[tileSize][maxLocalColumn];
    __local float4 submatrixB[tileSize][maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[localRow][v];
            const float4 vectorA1 = submatrixA[localRow + 1][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
}

__kernel void multiplyMatricesTile64V4H4(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 4;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 64;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[tileSize][maxLocalColumn];
    __local float4 submatrixB[tileSize][maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    float4 resultElements2 = (float4) (0.0f);
    float4 resultElements3 = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 2][localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 3][localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 2][localColumn] = b[(tiledRow + 2) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 3][localColumn] = b[(tiledRow + 3) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[localRow][v];
            const float4 vectorA1 = submatrixA[localRow + 1][v];
            const float4 vectorA2 = submatrixA[localRow + 2][v];
            const float4 vectorA3 = submatrixA[localRow + 3][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements2 += vectorA2.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements3 += vectorA3.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements2 += vectorA2.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements3 += vectorA3.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements2 += vectorA2.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements3 += vectorA3.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements2 += vectorA2.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements3 += vectorA3.s3 * submatrixB[vectorWidth * v + 3][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
    result[(globalRow + 2) * nDimension / vectorWidth + globalColumn] = resultElements2;
    result[(globalRow + 3) * nDimension / vectorWidth + globalColumn] = resultElements3;
}

__kernel void multiplyMatricesTile64V4H8(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 8;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 64;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[tileSize][maxLocalColumn];
    __local float4 submatrixB[tileSize][maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    float4 resultElements2 = (float4) (0.0f);
    float4 resultElements3 = (float4) (0.0f);
    float4 resultElements4 = (float4) (0.0f);
    float4 resultElements5 = (float4) (0.0f);
    float4 resultElements6 = (float4) (0.0f);
    float4 resultElements7 = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 2][localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 3][localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 4][localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 5][localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 6][localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 7][localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 2][localColumn] = b[(tiledRow + 2) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 3][localColumn] = b[(tiledRow + 3) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 4][localColumn] = b[(tiledRow + 4) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 5][localColumn] = b[(tiledRow + 5) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 6][localColumn] = b[(tiledRow + 6) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 7][localColumn] = b[(tiledRow + 7) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[localRow][v];
            const float4 vectorA1 = submatrixA[localRow + 1][v];
            const float4 vectorA2 = submatrixA[localRow + 2][v];
            const float4 vectorA3 = submatrixA[localRow + 3][v];
            const float4 vectorA4 = submatrixA[localRow + 4][v];
            const float4 vectorA5 = submatrixA[localRow + 5][v];
            const float4 vectorA6 = submatrixA[localRow + 6][v];
            const float4 vectorA7 = submatrixA[localRow + 7][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements2 += vectorA2.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements3 += vectorA3.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements4 += vectorA4.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements5 += vectorA5.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements6 += vectorA6.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements7 += vectorA7.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements2 += vectorA2.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements3 += vectorA3.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements4 += vectorA4.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements5 += vectorA5.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements6 += vectorA6.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements7 += vectorA7.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements2 += vectorA2.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements3 += vectorA3.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements4 += vectorA4.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements5 += vectorA5.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements6 += vectorA6.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements7 += vectorA7.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements2 += vectorA2.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements3 += vectorA3.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements4 += vectorA4.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements5 += vectorA5.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements6 += vectorA6.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements7 += vectorA7.s3 * submatrixB[vectorWidth * v + 3][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
    result[(globalRow + 2) * nDimension / vectorWidth + globalColumn] = resultElements2;
    result[(globalRow + 3) * nDimension / vectorWidth + globalColumn] = resultElements3;
    result[(globalRow + 4) * nDimension / vectorWidth + globalColumn] = resultElements4;
    result[(globalRow + 5) * nDimension / vectorWidth + globalColumn] = resultElements5;
    result[(globalRow + 6) * nDimension / vectorWidth + globalColumn] = resultElements6;
    result[(globalRow + 7) * nDimension / vectorWidth + globalColumn] = resultElements7;
}

__kernel void multiplyMatricesTile64V4H16(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 16;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 64;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[tileSize][maxLocalColumn];
    __local float4 submatrixB[tileSize][maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    float4 resultElements2 = (float4) (0.0f);
    float4 resultElements3 = (float4) (0.0f);
    float4 resultElements4 = (float4) (0.0f);
    float4 resultElements5 = (float4) (0.0f);
    float4 resultElements6 = (float4) (0.0f);
    float4 resultElements7 = (float4) (0.0f);
    float4 resultElements8 = (float4) (0.0f);
    float4 resultElements9 = (float4) (0.0f);
    float4 resultElements10 = (float4) (0.0f);
    float4 resultElements11 = (float4) (0.0f);
    float4 resultElements12 = (float4) (0.0f);
    float4 resultElements13 = (float4) (0.0f);
    float4 resultElements14 = (float4) (0.0f);
    float4 resultElements15 = (float4) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 2][localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 3][localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 4][localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 5][localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 6][localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 7][localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 8][localColumn] = a[(globalRow + 8) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 9][localColumn] = a[(globalRow + 9) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 10][localColumn] = a[(globalRow + 10) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 11][localColumn] = a[(globalRow + 11) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 12][localColumn] = a[(globalRow + 12) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 13][localColumn] = a[(globalRow + 13) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 14][localColumn] = a[(globalRow + 14) * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 15][localColumn] = a[(globalRow + 15) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 2][localColumn] = b[(tiledRow + 2) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 3][localColumn] = b[(tiledRow + 3) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 4][localColumn] = b[(tiledRow + 4) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 5][localColumn] = b[(tiledRow + 5) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 6][localColumn] = b[(tiledRow + 6) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 7][localColumn] = b[(tiledRow + 7) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 8][localColumn] = b[(tiledRow + 8) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 9][localColumn] = b[(tiledRow + 9) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 10][localColumn] = b[(tiledRow + 10) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 11][localColumn] = b[(tiledRow + 11) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 12][localColumn] = b[(tiledRow + 12) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 13][localColumn] = b[(tiledRow + 13) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 14][localColumn] = b[(tiledRow + 14) * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 15][localColumn] = b[(tiledRow + 15) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[localRow][v];
            const float4 vectorA1 = submatrixA[localRow + 1][v];
            const float4 vectorA2 = submatrixA[localRow + 2][v];
            const float4 vectorA3 = submatrixA[localRow + 3][v];
            const float4 vectorA4 = submatrixA[localRow + 4][v];
            const float4 vectorA5 = submatrixA[localRow + 5][v];
            const float4 vectorA6 = submatrixA[localRow + 6][v];
            const float4 vectorA7 = submatrixA[localRow + 7][v];
            const float4 vectorA8 = submatrixA[localRow + 8][v];
            const float4 vectorA9 = submatrixA[localRow + 9][v];
            const float4 vectorA10 = submatrixA[localRow + 10][v];
            const float4 vectorA11 = submatrixA[localRow + 11][v];
            const float4 vectorA12 = submatrixA[localRow + 12][v];
            const float4 vectorA13 = submatrixA[localRow + 13][v];
            const float4 vectorA14 = submatrixA[localRow + 14][v];
            const float4 vectorA15 = submatrixA[localRow + 15][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements2 += vectorA2.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements3 += vectorA3.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements4 += vectorA4.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements5 += vectorA5.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements6 += vectorA6.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements7 += vectorA7.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements8 += vectorA8.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements9 += vectorA9.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements10 += vectorA10.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements11 += vectorA11.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements12 += vectorA12.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements13 += vectorA13.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements14 += vectorA14.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements15 += vectorA15.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements2 += vectorA2.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements3 += vectorA3.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements4 += vectorA4.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements5 += vectorA5.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements6 += vectorA6.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements7 += vectorA7.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements8 += vectorA8.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements9 += vectorA9.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements10 += vectorA10.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements11 += vectorA11.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements12 += vectorA12.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements13 += vectorA13.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements14 += vectorA14.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements15 += vectorA15.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements2 += vectorA2.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements3 += vectorA3.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements4 += vectorA4.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements5 += vectorA5.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements6 += vectorA6.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements7 += vectorA7.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements8 += vectorA8.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements9 += vectorA9.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements10 += vectorA10.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements11 += vectorA11.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements12 += vectorA12.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements13 += vectorA13.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements14 += vectorA14.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements15 += vectorA15.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements2 += vectorA2.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements3 += vectorA3.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements4 += vectorA4.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements5 += vectorA5.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements6 += vectorA6.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements7 += vectorA7.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements8 += vectorA8.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements9 += vectorA9.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements10 += vectorA10.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements11 += vectorA11.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements12 += vectorA12.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements13 += vectorA13.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements14 += vectorA14.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements15 += vectorA15.s3 * submatrixB[vectorWidth * v + 3][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
    result[(globalRow + 2) * nDimension / vectorWidth + globalColumn] = resultElements2;
    result[(globalRow + 3) * nDimension / vectorWidth + globalColumn] = resultElements3;
    result[(globalRow + 4) * nDimension / vectorWidth + globalColumn] = resultElements4;
    result[(globalRow + 5) * nDimension / vectorWidth + globalColumn] = resultElements5;
    result[(globalRow + 6) * nDimension / vectorWidth + globalColumn] = resultElements6;
    result[(globalRow + 7) * nDimension / vectorWidth + globalColumn] = resultElements7;
    result[(globalRow + 8) * nDimension / vectorWidth + globalColumn] = resultElements8;
    result[(globalRow + 9) * nDimension / vectorWidth + globalColumn] = resultElements9;
    result[(globalRow + 10) * nDimension / vectorWidth + globalColumn] = resultElements10;
    result[(globalRow + 11) * nDimension / vectorWidth + globalColumn] = resultElements11;
    result[(globalRow + 12) * nDimension / vectorWidth + globalColumn] = resultElements12;
    result[(globalRow + 13) * nDimension / vectorWidth + globalColumn] = resultElements13;
    result[(globalRow + 14) * nDimension / vectorWidth + globalColumn] = resultElements14;
    result[(globalRow + 15) * nDimension / vectorWidth + globalColumn] = resultElements15;
}

__kernel void multiplyMatricesTile64V8(
        __global const float8* a,
        __global const float8* b,
        __global float8* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int vectorWidth = 8;
    const unsigned int tileSize = 64;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localRow = get_local_id(1);
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + localRow;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float8 submatrixA[tileSize][maxLocalColumn];
    __local float8 submatrixB[tileSize][maxLocalColumn];
    float8 resultElements = (float8) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float8 vectorA = submatrixA[localRow][v];
            resultElements += vectorA.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements += vectorA.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements += vectorA.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements += vectorA.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements += vectorA.s4 * submatrixB[vectorWidth * v + 4][localColumn];
            resultElements += vectorA.s5 * submatrixB[vectorWidth * v + 5][localColumn];
            resultElements += vectorA.s6 * submatrixB[vectorWidth * v + 6][localColumn];
            resultElements += vectorA.s7 * submatrixB[vectorWidth * v + 7][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements;
}

__kernel void multiplyMatricesTile64V8H2(
        __global const float8* a,
        __global const float8* b,
        __global float8* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 2;
    const unsigned int vectorWidth = 8;
    const unsigned int tileSize = 64;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float8 submatrixA[tileSize][maxLocalColumn];
    __local float8 submatrixB[tileSize][maxLocalColumn];
    float8 resultElements0 = (float8) (0.0f);
    float8 resultElements1 = (float8) (0.0f);
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
        submatrixA[localRow][localColumn] = a[globalRow * kDimension / vectorWidth + tiledColumn];
        submatrixA[localRow + 1][localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
        submatrixB[localRow][localColumn] = b[tiledRow * nDimension / vectorWidth + globalColumn];
        submatrixB[localRow + 1][localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float8 vectorA0 = submatrixA[localRow][v];
            const float8 vectorA1 = submatrixA[localRow + 1][v];
            resultElements0 += vectorA0.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[vectorWidth * v][localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[vectorWidth * v + 1][localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[vectorWidth * v + 2][localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[vectorWidth * v + 3][localColumn];
            resultElements0 += vectorA0.s4 * submatrixB[vectorWidth * v + 4][localColumn];
            resultElements1 += vectorA1.s4 * submatrixB[vectorWidth * v + 4][localColumn];
            resultElements0 += vectorA0.s5 * submatrixB[vectorWidth * v + 5][localColumn];
            resultElements1 += vectorA1.s5 * submatrixB[vectorWidth * v + 5][localColumn];
            resultElements0 += vectorA0.s6 * submatrixB[vectorWidth * v + 6][localColumn];
            resultElements1 += vectorA1.s6 * submatrixB[vectorWidth * v + 6][localColumn];
            resultElements0 += vectorA0.s7 * submatrixB[vectorWidth * v + 7][localColumn];
            resultElements1 += vectorA1.s7 * submatrixB[vectorWidth * v + 7][localColumn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[globalRow * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
}

__kernel void multiplyMatricesTile64V4H8P(
        __global const float4* a,
        __global const float4* b,
        __global float4* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = 8;
    const unsigned int vectorWidth = 4;
    const unsigned int tileSize = 64;
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local float4 submatrixA[2][tileSize * maxLocalColumn];
    __local float4 submatrixB[2][tileSize * maxLocalColumn];
    float4 resultElements0 = (float4) (0.0f);
    float4 resultElements1 = (float4) (0.0f);
    float4 resultElements2 = (float4) (0.0f);
    float4 resultElements3 = (float4) (0.0f);
    float4 resultElements4 = (float4) (0.0f);
    float4 resultElements5 = (float4) (0.0f);
    float4 resultElements6 = (float4) (0.0f);
    float4 resultElements7 = (float4) (0.0f);
    submatrixA[0][(localRow + 0) * maxLocalColumn + localColumn] = a[(globalRow + 0) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 1) * maxLocalColumn + localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 2) * maxLocalColumn + localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 3) * maxLocalColumn + localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 4) * maxLocalColumn + localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 5) * maxLocalColumn + localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 6) * maxLocalColumn + localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + localColumn];
    submatrixA[0][(localRow + 7) * maxLocalColumn + localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + localColumn];
    submatrixB[0][(localRow + 0) * maxLocalColumn + localColumn] = b[(localRow + 0) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 1) * maxLocalColumn + localColumn] = b[(localRow + 1) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 2) * maxLocalColumn + localColumn] = b[(localRow + 2) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 3) * maxLocalColumn + localColumn] = b[(localRow + 3) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 4) * maxLocalColumn + localColumn] = b[(localRow + 4) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 5) * maxLocalColumn + localColumn] = b[(localRow + 5) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 6) * maxLocalColumn + localColumn] = b[(localRow + 6) * nDimension / vectorWidth + globalColumn];
    submatrixB[0][(localRow + 7) * maxLocalColumn + localColumn] = b[(localRow + 7) * nDimension / vectorWidth + globalColumn];
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int nextTile = tile + 1;
        if (nextTile < numberOfTiles) {
            const unsigned int tiledRow = tileSize * nextTile + localRow;
            const unsigned int tiledColumn = maxLocalColumn * nextTile + localColumn;
            submatrixA[nextTile % 2][(localRow + 0) * maxLocalColumn + localColumn] = a[(globalRow + 0) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 1) * maxLocalColumn + localColumn] = a[(globalRow + 1) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 2) * maxLocalColumn + localColumn] = a[(globalRow + 2) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 3) * maxLocalColumn + localColumn] = a[(globalRow + 3) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 4) * maxLocalColumn + localColumn] = a[(globalRow + 4) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 5) * maxLocalColumn + localColumn] = a[(globalRow + 5) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 6) * maxLocalColumn + localColumn] = a[(globalRow + 6) * kDimension / vectorWidth + tiledColumn];
            submatrixA[nextTile % 2][(localRow + 7) * maxLocalColumn + localColumn] = a[(globalRow + 7) * kDimension / vectorWidth + tiledColumn];
            submatrixB[nextTile % 2][(localRow + 0) * maxLocalColumn + localColumn] = b[(tiledRow + 0) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 1) * maxLocalColumn + localColumn] = b[(tiledRow + 1) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 2) * maxLocalColumn + localColumn] = b[(tiledRow + 2) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 3) * maxLocalColumn + localColumn] = b[(tiledRow + 3) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 4) * maxLocalColumn + localColumn] = b[(tiledRow + 4) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 5) * maxLocalColumn + localColumn] = b[(tiledRow + 5) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 6) * maxLocalColumn + localColumn] = b[(tiledRow + 6) * nDimension / vectorWidth + globalColumn];
            submatrixB[nextTile % 2][(localRow + 7) * maxLocalColumn + localColumn] = b[(tiledRow + 7) * nDimension / vectorWidth + globalColumn];
        }
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
            const float4 vectorA0 = submatrixA[tile % 2][(localRow + 0) * maxLocalColumn + v];
            const float4 vectorA1 = submatrixA[tile % 2][(localRow + 1) * maxLocalColumn + v];
            const float4 vectorA2 = submatrixA[tile % 2][(localRow + 2) * maxLocalColumn + v];
            const float4 vectorA3 = submatrixA[tile % 2][(localRow + 3) * maxLocalColumn + v];
            const float4 vectorA4 = submatrixA[tile % 2][(localRow + 4) * maxLocalColumn + v];
            const float4 vectorA5 = submatrixA[tile % 2][(localRow + 5) * maxLocalColumn + v];
            const float4 vectorA6 = submatrixA[tile % 2][(localRow + 6) * maxLocalColumn + v];
            const float4 vectorA7 = submatrixA[tile % 2][(localRow + 7) * maxLocalColumn + v];
            resultElements0 += vectorA0.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s0 * submatrixB[tile % 2][(vectorWidth * v + 0) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s1 * submatrixB[tile % 2][(vectorWidth * v + 1) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s2 * submatrixB[tile % 2][(vectorWidth * v + 2) * maxLocalColumn + localColumn];
            resultElements0 += vectorA0.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements1 += vectorA1.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements2 += vectorA2.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements3 += vectorA3.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements4 += vectorA4.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements5 += vectorA5.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements6 += vectorA6.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
            resultElements7 += vectorA7.s3 * submatrixB[tile % 2][(vectorWidth * v + 3) * maxLocalColumn + localColumn];
        }
    }
    result[(globalRow + 0) * nDimension / vectorWidth + globalColumn] = resultElements0;
    result[(globalRow + 1) * nDimension / vectorWidth + globalColumn] = resultElements1;
    result[(globalRow + 2) * nDimension / vectorWidth + globalColumn] = resultElements2;
    result[(globalRow + 3) * nDimension / vectorWidth + globalColumn] = resultElements3;
    result[(globalRow + 4) * nDimension / vectorWidth + globalColumn] = resultElements4;
    result[(globalRow + 5) * nDimension / vectorWidth + globalColumn] = resultElements5;
    result[(globalRow + 6) * nDimension / vectorWidth + globalColumn] = resultElements6;
    result[(globalRow + 7) * nDimension / vectorWidth + globalColumn] = resultElements7;
}
