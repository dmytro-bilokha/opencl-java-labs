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
