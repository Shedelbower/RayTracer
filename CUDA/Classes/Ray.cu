#include <math.h>
#include "Vector3.cuh"

// Default Constructor
__device__ Ray::Ray() {
    origin = Vector3();
    direction = Vector3();
}

// Main Constructor
__device__ Ray::Ray(const Vector3& origin, const Vector3& direction) {
    this->origin = Vector3(origin);
    this->direction = Vector3(direction);
}

// Copy Constructor
__device__ Ray::Ray(const Ray& other) {
    this->origin = Vector3(other.origin);
    this->direction = Vector3(other.direction);
}

// Deconstructor
__device__ Ray::~Ray() {
    // delete origin;
    // delete direction;
}


__device__ Vector3 Ray::getPoint(float t) const {
    Vector3 point = Vector3::multiply(this->origin, t);
    return point;
}
