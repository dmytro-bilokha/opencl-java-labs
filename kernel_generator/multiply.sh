#!/bin/sh -

print_kernel() {
    cat <<EOF
__kernel void multiplyMatricesTile${tileSize}V${vectorWidth}H${workHeight}(
        __global const ${floatType}* a,
        __global const ${floatType}* b,
        __global ${floatType}* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = ${workHeight};
    const unsigned int vectorWidth = ${vectorWidth};
    const unsigned int tileSize = ${tileSize};
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local ${floatType} submatrixA[tileSize][maxLocalColumn];
    __local ${floatType} submatrixB[tileSize][maxLocalColumn];
EOF
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "    ${floatType} resultElements${i} = (${floatType}) (0.0f);"
        i=$((${i}+1))
    done
    cat <<EOF
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        const unsigned int tiledRow = tileSize * tile + localRow;
        const unsigned int tiledColumn = maxLocalColumn * tile + localColumn;
EOF
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "        submatrixA[localRow + ${i}][localColumn] = a[(globalRow + ${i}) * kDimension / vectorWidth + tiledColumn];"
        i=$((${i}+1))
    done
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "        submatrixB[localRow + ${i}][localColumn] = b[(tiledRow + ${i}) * nDimension / vectorWidth + globalColumn];"
        i=$((${i}+1))
    done
    cat <<EOF
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
EOF
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "            const ${floatType} vectorA${i} = submatrixA[localRow + ${i}][v];"
        i=$((${i}+1))
    done
    j=0
    while [ ${j} -ne ${vectorWidth} ]; do
        i=0
        while [ ${i} -ne ${workHeight} ]; do
            echo "            resultElements${i} += vectorA${i}.s${j} * submatrixB[vectorWidth * v + ${j}][localColumn];"
            i=$((${i}+1))
        done
        j=$((${j}+1))
    done
    cat <<EOF
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
EOF
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "    result[(globalRow + ${i}) * nDimension / vectorWidth + globalColumn] = resultElements${i};"
        i=$((${i}+1))
    done
    echo "}"
}

print_kernel_prefetch() {
    cat <<EOF
__kernel void multiplyMatricesTile${tileSize}V${vectorWidth}H${workHeight}P(
        __global const ${floatType}* a,
        __global const ${floatType}* b,
        __global ${floatType}* result,
        const unsigned long mDimension,
        const unsigned long kDimension,
        const unsigned long nDimension
) {
    const unsigned int workHeight = ${workHeight};
    const unsigned int vectorWidth = ${vectorWidth};
    const unsigned int tileSize = ${tileSize};
    const unsigned int maxLocalColumn = tileSize / vectorWidth;
    const unsigned int localColumn = get_local_id(0);
    const unsigned int globalRow = tileSize * get_group_id(1) + get_local_id(1) * workHeight;
    const unsigned int localRow = globalRow % tileSize;
    const unsigned int globalColumn = maxLocalColumn * get_group_id(0) + localColumn;
    __local ${floatType} submatrixA[2][tileSize * maxLocalColumn];
    __local ${floatType} submatrixB[2][tileSize * maxLocalColumn];
EOF
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "    ${floatType} resultElements${i} = (${floatType}) (0.0f);"
        i=$((${i}+1))
    done
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "    submatrixA[0][(localRow + ${i}) * maxLocalColumn + localColumn] = a[(globalRow + ${i}) * kDimension / vectorWidth + localColumn];"
        i=$((${i}+1))
    done
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "    submatrixB[0][(localRow + ${i}) * maxLocalColumn + localColumn] = b[(localRow + ${i}) * nDimension / vectorWidth + globalColumn];"
        i=$((${i}+1))
    done
    cat <<EOF
    const unsigned int numberOfTiles = kDimension / tileSize;
    for (unsigned int tile = 0; tile < numberOfTiles; tile++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int nextTile = tile + 1;
        if (nextTile < numberOfTiles) {
            const unsigned int tiledRow = tileSize * nextTile + localRow;
            const unsigned int tiledColumn = maxLocalColumn * nextTile + localColumn;
EOF
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "            submatrixA[nextTile % 2][(localRow + ${i}) * maxLocalColumn + localColumn] = a[(globalRow + ${i}) * kDimension / vectorWidth + tiledColumn];"
        i=$((${i}+1))
    done
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "            submatrixB[nextTile % 2][(localRow + ${i}) * maxLocalColumn + localColumn] = b[(tiledRow + ${i}) * nDimension / vectorWidth + globalColumn];"
        i=$((${i}+1))
    done
    cat <<EOF
        }
        for (unsigned int v = 0; v < maxLocalColumn; v++) {
EOF
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "            const ${floatType} vectorA${i} = submatrixA[tile % 2][(localRow + ${i}) * maxLocalColumn + v];"
        i=$((${i}+1))
    done
    j=0
    while [ ${j} -ne ${vectorWidth} ]; do
        i=0
        while [ ${i} -ne ${workHeight} ]; do
            echo "            resultElements${i} += vectorA${i}.s${j} * submatrixB[tile % 2][(vectorWidth * v + ${j}) * maxLocalColumn + localColumn];"
            i=$((${i}+1))
        done
        j=$((${j}+1))
    done
    cat <<EOF
        }
    }
EOF
    i=0
    while [ ${i} -ne ${workHeight} ]; do
        echo "    result[(globalRow + ${i}) * nDimension / vectorWidth + globalColumn] = resultElements${i};"
        i=$((${i}+1))
    done
    echo "}"
}

if [ "$#" -lt 3 ]; then
    echo "Error: At least 3 arguments required" >&2
    echo "Usage: ${0} tileSize vectorWidth workHeight [prefetch]" >&2
    exit 1
fi
tileSize="${1}"
vectorWidth="${2}"
workHeight="${3}"
floatType="float${vectorWidth}"
if [ ${vectorWidth} -eq 1 ]; then
    floatType="float"
fi

if [ $# -eq 4 ]; then
    print_kernel_prefetch
else
    print_kernel
fi
