#ifndef AABB_H
#define AABB_H

#include <math.h>
#include "Vector3.cuh"
//#include "Primitive.cuh"

class AABB {

public:

    Vector3 pMin, pMax;
    
    // TODO
    // __device__ Vector3 getNormal(const Vector3& point) const {
    //     Vector3 normal = Vector3(-1.0,-1.0,-1.0);
    //     return normal;
    // }

    __device__ static void orderValues(float * tMin, float * tMax) {
        if (tMin[0] > tMax[0]) {
            float temp = tMin[0];
            tMin[0] = tMax[0];
            tMax[0] = temp;
        }
    }

    // TODO
    __device__ float raycast(const Ray& ray) const {
        
        Vector3 o = ray.origin;
        Vector3 d = ray.direction;
        
        // A small number is added to the direction to try to prevent divide by 0 errors
        float offset = 0.000001; //TODO: Check if this is needed in C++
        float tMin = (pMin.x - o.x) / (d.x + offset);
        float tMax = (pMax.x - o.x) / (d.x + offset);
        float tYMin = (pMin.y - o.y) / (d.y + offset);
        float tYMax = (pMax.y - o.y) / (d.y + offset);
        float tZMin = (pMin.z - o.z) / (d.z + offset);
        float tZMax = (pMax.z - o.z) / (d.z + offset);

        
        // Order values so min < max
        orderValues(&tMin, &tMax);
        orderValues(&tYMin, &tYMax);

        if (tMin > tYMax || tYMin > tMax) {
            return -666.0;
        }

        tMin = max(tMin, tYMin);
        tMax = min(tMax, tYMax);

        orderValues(&tZMin, &tZMax);

        if (tMin > tZMax || tZMin > tMax) {
            return -666.0;
        }

        tMin = max(tMin, tZMin);
        tMax = min(tMax, tZMax);

        if (tMax < 0) {
            // Box is behind ray
            return - 666.0;
        }

        // RAY INTERSECTS WITH THE BOX!

        if (tMin < 0) {
            // Ray origin is inside the box
            return 0.0;
        }

        return tMin; // Set the t output value to the value where the ray enters the box.
        
    }


    __device__ bool rayIntersects(const Ray& ray) const {
        float distance = this->raycast(ray);
        return distance >= 0;
    }


    /*
    __device__ float raycast(const Ray& ray, Vector3*& hitPoint) const {
        float t = this->raycast(ray);
        if (t >= 0) {
            *hitPoint = ray.getPoint(t);
        } else {
            hitPoint = NULL; // Ray did not hit anything, make hit point NULL
        }
        return t;
    }

    __device__ float raycast(const Ray& ray, Vector3*& hitPoint, Vector3*& normal) const {
        float t = this->raycast(ray, hitPoint);

        // Check if ray didn't hit primitive
        if (hitPoint == NULL) {
            normal = NULL;
            return t;
        }

        *normal = this->getNormal(*hitPoint);
        return t;
    }

    */

};


#endif // AABB_H