#ifndef BVHLEAFNODE_H
#define BVHLEAFNODE_H

#include "Primitive.cuh"
#include "AABB.cuh"
#include "BVHNode.cuh"

class BVHLeafNode: public BVHNode {

public:

    float primitiveCount;

    __device__ int getPrimitiveCount() {
        return (int) this->primitiveCount;
    }

    __device__ AABB * getPrimitiveAABB(int index) {
        float * ptr = (float*) this;
        float * firstAABB = ptr + 9;
        float * AABBf = firstAABB + 7*index;
        return (AABB*)AABBf;
    }

    __device__ int getPrimitiveIndex(int index) {
        float * ptr = (float*) this->getPrimitiveAABB(index);
        ptr += 6;
        return (int) *ptr;
    }

    // __device__ int getPrimitiveType(int index) {
    //     float * ptr = (float*) this->getPrimitiveAABB(index);
    //     ptr += sizeof(AABB);
    //     return (int) *ptr;
    // }

    __device__ void checkPrimitives(Ray * ray, Primitive * primitives, int * primitiveIndex, float * distance) {
    
        int primitiveCount = this->getPrimitiveCount();


        // int primitiveCount = getNodePrimitiveCount(node);
        // float * primitiveBoxes = getNodePrimitiveBoxes(node);

        float tMin = -1.0;
        int minIndex = -1;

        for (int i = 0; i < primitiveCount; i++) {
            AABB * primitiveBox = this->getPrimitiveAABB(i);
            int index = this->getPrimitiveIndex(i);
            Primitive * primitive = (Primitive*) (((float*)primitives) + index);
            float currT = -1.0;

            currT = primitive->raycast(*ray);

            if (currT > 0 && (currT < tMin || tMin < 0)) {
                tMin = currT;
                minIndex = index;
            }
        }

        *distance = tMin;
        *primitiveIndex = minIndex;
    }

};



#endif //BVHLEAFNODE_H