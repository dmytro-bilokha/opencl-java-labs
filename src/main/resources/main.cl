__kernel void simple_add(__global const float* A, __global const float* B, __global float* C) {
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
}

__kernel void addMatrices(__global const float* a, __global const float* b, __global float* result, const unsigned long elementCount) {
    int id = get_global_id(0);
    int globalSize = get_global_size(0);
    for (int i = id; i < elementCount; i += globalSize) {
        result[i] = a[i] + b[i];
    }
}

__kernel void addMatricesLined(__global const float* a, __global const float* b, __global float* result, const unsigned long elementCount) {
    int id = get_global_id(0);
    int globalSize = get_global_size(0);
    int elementsPerWorker = elementCount / globalSize + 1;
    int startIndex = elementsPerWorker * id;
    int endIndex = startIndex + elementsPerWorker;
    if (endIndex > elementCount) {
        endIndex = elementCount;
    }
    for (int i = startIndex; i < endIndex; i++) {
        result[i] = a[i] + b[i];
    }
}

__kernel void addMatricesTwo(__global const float* a, __global const float* b, __global float* result, const unsigned long elementCount) {
    int id = get_global_id(0);
    int globalSize = get_global_size(0);
    int step = globalSize * 2;
    int maxI = elementCount - 1;
    int i = id * 2;
    for (; i < maxI; i += step) {
        result[i] = a[i] + b[i];
        result[i + 1] = a[i + 1] + b[i + 1];
    }
    if (i == maxI) {
        result[i] = a[i] + b[i];
    }
}
