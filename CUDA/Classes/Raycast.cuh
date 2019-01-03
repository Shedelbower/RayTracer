#ifndef RAYCAST_H
#define RAYCAST_H

#include <math.h>
#include "Vector3.cuh"
#include "Ray.cuh"

#include "Primitive.cuh"
#include "Sphere.cuh"
#include "AABB.cuh"

#include "BVHNode.cuh"
#include "BVHInternalNode.cuh"
#include "BVHLeafNode.cuh"

class Raycast {

public:

    // STATIC
    __device__ static void raycast(Ray * ray, BVHNode * bvh, Primitive * primitives,
                        float * distance, Primitive ** primitive, Vector3 * normal, Vector3 * hitPoint // Output
                        ) {

        int stackSize = 21;
        BVHNode ** stack = (BVHNode**) malloc(stackSize * sizeof(BVHNode*));

        int stackIndex = 0;
        
        stack[stackIndex] = bvh; // Start with root node on stack

        float minDist = -1.0;
        int minIndex = -1;

        while (stackIndex >= 0) {

            // Pop top node off of stack
            BVHNode * node = stack[stackIndex];
            stackIndex--;

            if (node->isLeafNode()) {
                float dist = -1.0;
                int primIndex = -1;

                BVHLeafNode * leaf = (BVHLeafNode*) node;
                leaf->checkPrimitives(ray, primitives, &primIndex, &dist);

                //TODO: Remove
                // if (dist > 0) {
                //     *distance += 0.1;
                // }
                
                // Update the minimum distance if applicable
                if (dist > 0 && (dist < minDist || minDist < 0)) {
                    minDist = dist;
                    minIndex = primIndex;

                }
            } else {
                BVHInternalNode * internal = (BVHInternalNode*) node;

                BVHNode * leftChild = internal->getLeftChild();
                BVHNode * rightChild = internal->getRightChild(); // TODO: Optimize by passing left child

                AABB * leftBox = &(leftChild->box);
                AABB * rightBox = &(rightChild->box);

                
                if (rightBox->rayIntersects(*ray)) {
                    // Add right child node to stack
                    stackIndex++;
                    if (stackIndex >= stackSize) {
                        *distance = 0.5; // TODO: Why did I do this again?
                        break;
                    }
                    stack[stackIndex] = rightChild;
                }

                if (leftBox->rayIntersects(*ray)) {
                    // Add left child node to stack
                    stackIndex++;
                    if (stackIndex >= stackSize) {
                        *distance = 0.5;
                        break;
                    }
                    stack[stackIndex] = leftChild;
                }
                
                
            }
        }

        free(stack);

        // Set output
        *distance = minDist;
        if (*distance >= 0) {
            *primitive = (Primitive*) (((float*)primitives) + minIndex);
            if (hitPoint != NULL) {
                *hitPoint = (*primitive)->getHitPoint(*ray, *distance);
            }
            if (normal != NULL) {
                *normal = (*primitive)->getNormal(*ray, *distance, *hitPoint);
            }
        }
    }


};

#endif // RAYCAST_H