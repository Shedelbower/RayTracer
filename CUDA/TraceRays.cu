#include <math.h>
#include "Vector3.cuh"
#include "Ray.cuh"
#include "RGBColor.cuh"
#include "Material.cuh"

#include "Primitive.cuh"
#include "Sphere.cuh"
#include "AABB.cuh"

#include "Light.cuh"

#include "BVHNode.cuh"
#include "BVHInternalNode.cuh"
#include "BVHLeafNode.cuh"

#include "Shader.cuh"
#include "Raycast.cuh"

#include "Image.cuh"




__device__ void traceRay(Ray * ray, BVHNode * bvh, Primitive * primitives,
                            Material * materials, Light * lights, mint lightCount,
                            RGBColor * backgroundColor, Image * skybox,
                            mint maxRecursionDepth,
                            mint enableShadows, mint enableSkybox,
                            RGBColor * color // Output
                        ) {



    // Allocate
    Vector3 * normal = new Vector3();
    Vector3 * hitPoint = new Vector3();

    for (int depth = maxRecursionDepth; depth > 0; depth--) {

        float distance = -1.0;
        Primitive * primitive = NULL;

        Raycast::raycast(ray, bvh, primitives,
                &distance, &primitive, normal, hitPoint // Output
        );

        //TODO: Remove
        // if (distance > 0) {
        //     *color = RGBColor(distance, distance, 0.0);
        //     break;
        // }



        Material * material = NULL;

        if (primitive != NULL) {
            material = primitive->getMaterial(materials);
        }
        
        *color = Shader::getColor(
            hitPoint, normal, *ray, material, bvh, primitives, materials, *backgroundColor, skybox, (enableSkybox > 0), (enableShadows > 0), lights, lightCount
            );

        float hitOffset = 0.001; // Should be big enough to avoid floating point errors


        if (material == NULL) {
            break;
        } else if (material->isPerfectMirror()) {
            // Reflect the ray across the normal for the next iteration.
            ray->origin = *hitPoint;
            ray->direction.reflect(*normal);

            // Move origin off of primitive's surface
            ray->origin.add(Vector3::multiply(*normal, hitOffset));
        } else if (material->isTransparent()) {
            // Figure out which side to offset point to
            int sign = 1.0;//??
            if (Vector3::dot(*normal, ray->direction) < 0) {
                sign = -1.0;
            }
            

            
            // Refract ray
            ray->origin = *hitPoint;
            ray->direction.refract(*normal, material->transparent);

            ray->origin.add(Vector3::multiply(*normal, hitOffset*sign));
        }


        // Show normals
        // *color = RGBColor(normal->x, normal->y, normal->z);

        



    }

    // Deallocate
    delete normal;
    delete hitPoint;

}


/*

float * rayOrigins, float * rayDirections,
                    float * primitives, float * materials, float * bvh,
                    float * lights,
                    float * backgroundColor, float * skybox, mint skyboxWidth, mint skyboxHeight,
                    mint rayCount, mint lightCount,
                    mint enableShadows, mint maxRecursionDepth,
                    float * colors
*/


__global__ void traceRays(Ray * rays, mint rayCount, BVHNode * bvh, Primitive * primitives,
                            Material * materials, Light * lights, mint lightCount,
                            RGBColor * backgroundColor, Image * skybox,
                            mint maxRecursionDepth,
                            mint enableShadows, mint enableSkybox,
                            RGBColor * colors) {
    
    int rayIndex = threadIdx.x + blockIdx.x*blockDim.x;

    if (rayIndex < rayCount) {

        Ray * ray = rays + rayIndex;
        RGBColor * color = colors + rayIndex; // Generate one pixel per ray

        // Triangle * triangle = (Triangle*)primitives;
        // float * value = (float*)triangle;
        // *color = RGBColor(0,0,*value);

        
        traceRay(ray, bvh, primitives,
                materials, lights, lightCount,
                backgroundColor, skybox, maxRecursionDepth,
                enableShadows, enableSkybox,
                color // Output
                );
        

    }
}



__global__ void testBVH(BVHNode * bvh, mint * internalCount, mint * leafCount) {

    *internalCount = 0;
    *leafCount = 0;

    int lCount = 0;
    int iCount = 0;

    int stackSize = 21;
    BVHNode ** stack = (BVHNode**) malloc(stackSize * sizeof(BVHNode*));

    int stackIndex = 0;

    
    stack[stackIndex] = bvh; // Start with root node on stack

    while (stackIndex >= 0) {

        // Pop top node off of stack
        BVHNode * node = stack[stackIndex];
        stackIndex--;

        if (node->isLeafNode()) {
            lCount += 1;
        } else {
            
            iCount += 1;

            BVHInternalNode * internal = (BVHInternalNode*) node;

            BVHNode * leftChild = internal->getLeftChild();
            BVHNode * rightChild = internal->getRightChild(); // TODO: Optimize by passing left child

            // Add Right Child
            stackIndex++;
            if (stackIndex >= stackSize) {
                lCount = -666;
                break;
            }
            stack[stackIndex] = rightChild;

            // Add left child
            stackIndex++;
            if (stackIndex >= stackSize) {
                lCount = -666;
                break;
            }

            stack[stackIndex] = leftChild;
        }
    }

    free(stack);

    *leafCount = lCount;
    *internalCount = iCount;
}
















