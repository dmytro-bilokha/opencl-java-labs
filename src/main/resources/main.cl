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
