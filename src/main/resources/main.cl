__kernel void simple_add(__global const float* A, __global const float* B, __global float* C) {
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
}

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

__kernel void multiplyMatricesNaive(
        __global const float* a,
        __global const float* b,
        __global float* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
        ) {
    const unsigned long globalRow = get_global_id(0);
    const unsigned long globalColumn = get_global_id(1);
    float resultElement = 0.0f;
    for (unsigned long k = 0; k < kDimension; k++) {
        resultElement += a[globalRow * kDimension + k] * b[k * nDimension + globalColumn];
    }
    result[globalRow * nDimension + globalColumn] = resultElement;
}

__kernel void multiplyMatricesNaiveO(
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
