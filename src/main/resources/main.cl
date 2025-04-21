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
