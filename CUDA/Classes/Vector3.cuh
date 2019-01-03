#ifndef VECTOR3_H
#define VECTOR3_H

class Vector3 {

public:
    float x, y, z;

    ////////// CONSTRUCTORS //////////

    // Default Constructor
    __device__ Vector3();

    // Main Constructor
    __device__ Vector3(float x, float y, float z);

    // Copy Constructor
    __device__ Vector3(const Vector3& other);

    // Deconstructor
    __device__ ~Vector3();


    ////////// MEMBER FUNCTIONS //////////

    /* MODIFY OBJECT */

    __device__ void normalize();

    __device__ void reverse();

    // Reflect this vector over the specified axis
    __device__ void reflect(const Vector3& axis);

    __device__ void refract(const Vector3& normal, float indexOfRefraction);


    /* CALCULATIONS */

    __device__ float magnitude() const;


    /* OPERATORS */

    __device__ void add(const float& value);

    __device__ void add(const Vector3& other);

    __device__ void subtract(const float& value);

    __device__ void subtract(const Vector3& other);

    __device__ void multiply(const float& value);

    __device__ void divide(const float& value);

    __device__ void divide(const Vector3& other);


    ////////// STATIC FUNCTIONS //////////

    __device__ static Vector3 add(const Vector3& a, const float& value);

    __device__ static Vector3 add(const Vector3& a, const Vector3& b);

    __device__ static Vector3 subtract(const Vector3& a, const Vector3& b);

    __device__ static Vector3 multiply(const Vector3& a, const float& value);

    __device__ static Vector3 multiply(const Vector3& a, const Vector3& b);

    __device__ static float dot(const Vector3& a, const Vector3& b);

    
};

#include "Vector3.cu"

#endif //VECTOR3_H