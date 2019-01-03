#ifndef BVHNODE_H
#define BVHNODE_H

#include "Primitive.cuh"
#include "AABB.cuh"

class BVHNode {

public:

    float isLeaf;
    float totalSize; // Size of this node and all of its children
    AABB box; // Bounding Box

    __device__ int size() {
        return (int) this->totalSize;
    }

    __device__ bool isLeafNode() {
        return this->isLeaf > 0;
    }
};



#endif //BVHNODE_H