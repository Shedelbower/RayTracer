#include "BVHNode.cuh"
#include "BVHInternalNode.cuh"
#include "BVHLeafNode.cuh"

#include "Primitive.cuh"
#include "Sphere.cuh"
#include "AABB.cuh"

__global__ void getBVHInfo(BVHNode * root, int * depth) {

}