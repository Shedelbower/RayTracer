#ifndef RAY_H
#define RAY_H

#include <math.h>
#include "Vector3.cuh"

class Ray {

    public:

    Vector3 origin, direction;
    
    ////////// CONSTRUCTORS //////////

    // Default Constructor
    __device__ Ray();

    // Parameterized Constructor
    __device__ Ray(const Vector3& origin, const Vector3& direction);

    // Copy Constructor
    __device__ Ray(const Ray& other);

    // Deconstructor
    __device__ ~Ray();

    ////////// MEMBER FUNCTIONS //////////

    __device__ Vector3 getPoint(float t) const;
};

#include "Ray.cu"

#endif // RAY_H