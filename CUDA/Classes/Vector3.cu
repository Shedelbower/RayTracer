#include <math.h>
#include "Vector3.cuh"

// Default Constructor
__device__ Vector3::Vector3() {
    x = 0.0;
    y = 0.0;
    z = 0.0;
}

__device__ Vector3::Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

__device__ Vector3::Vector3(const Vector3& other) : x(other.x), y(other.y), z(other.z) {}

__device__ Vector3::~Vector3() {
    // No cleanup required
}


__device__ void Vector3::normalize() {
    float magnitude = this->magnitude();
    x /= magnitude;
    y /= magnitude;
    z /= magnitude;
}

__device__ void Vector3::reverse() {
    x *= -1;
    y *= -1;
    z *= -1;
}

// Reflect this vector over the specified axis
__device__ void Vector3::reflect(const Vector3& axis) {
    float dot = Vector3::dot(*this, axis);
    Vector3 vec = Vector3(axis);
    vec.multiply(dot * 2);
    this->subtract(vec);
}

__device__ void Vector3::refract(const Vector3& normal, float indexOfRefraction) {

    float dot = Vector3::dot(*this, normal);

    if (dot < -1) {
        dot = -1.0;
    } else if (dot > 1) {
        dot = 1.0;
    }

    float etai = 1.0; //Air's index of refraction
    float etat = indexOfRefraction;

    float cosi = dot;
    float normalCoef = 1.0;
    if (cosi < 0) {
        cosi = -cosi;
    } else {
        // Swap etai and etat
        float temp = etai;
        etai = etat;
        etat = temp;

        // Reverse normal
        normalCoef = -1.0;
    }

    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);

    if (k < 0) {
        x = 0;
        y = 0;
        z = 0;
    } else {
        normalCoef *= (eta * cosi - sqrtf(k));
        this->multiply(eta);
        x += normal.x * normalCoef;
        y += normal.y * normalCoef;
        z += normal.z * normalCoef;
    }
}

// Info Functions

__device__ float Vector3::magnitude() const {
    float radicand = dot(*this, *this);
    return sqrtf(radicand);
}

// Operator Functions

__device__ void Vector3::add(const float& value) {
    x += value;
    y += value;
    z += value;
}

__device__ void Vector3::add(const Vector3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
}

__device__ void Vector3::subtract(const float& value) {
    x -= value;
    y -= value;
    z -= value;
}

__device__ void Vector3::subtract(const Vector3& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
}

__device__ void Vector3::multiply(const float& value) {
    x *= value;
    y *= value;
    z *= value;
}

__device__ void Vector3::divide(const float& value) {
    x /= value;
    y /= value;
    z /= value;
}

__device__ void Vector3::divide(const Vector3& other) {
    x /= other.x;
    y /= other.y;
    z /= other.z;
}


// STATIC FUNCTIONS

__device__ Vector3 Vector3::add(const Vector3& a, const float& value) {
    Vector3 sum = Vector3(a);
    sum.add(value);
    return sum;
}

__device__ Vector3 Vector3::add(const Vector3& a, const Vector3& b) {
    Vector3 sum = Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
    return sum;
}

__device__ Vector3 Vector3::subtract(const Vector3& a, const Vector3& b) {
    Vector3 difference = Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
    return difference;
}

__device__ Vector3 Vector3::multiply(const Vector3& a, const float& value) {
    Vector3 product = Vector3(a);
    product.multiply(value);
    return product;
}

__device__ Vector3 Vector3::multiply(const Vector3& a, const Vector3& b) {
    Vector3 product = Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
    return product; 
}

__device__ float Vector3::dot(const Vector3& a, const Vector3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


