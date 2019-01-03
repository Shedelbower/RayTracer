#ifndef BVHINTERNALNODE_H
#define BVHINTERNALNODE_H

#include "Primitive.cuh"
#include "AABB.cuh"
#include "BVHNode.cuh"

class BVHInternalNode: public BVHNode {

public:

    __device__ BVHNode * getLeftChild() {
        float * ptr = (float*) this;
        BVHNode * child = (BVHNode*)(ptr + 8);
        return child;
    }

    __device__ BVHNode * getRightChild() {
        BVHNode * leftChild = this->getLeftChild();
        float * ptr = (float*) leftChild;
        BVHNode * rightChild = (BVHNode*) (ptr + leftChild->size());
        return rightChild;
    }

};



#endif //BVHINTERNALNODE_H