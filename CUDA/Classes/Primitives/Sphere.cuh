#ifndef SPHERE_H
#define SPHERE_H

#include <math.h>
#include "Vector3.cuh"
#include "Primitive.cuh"


class Sphere: public Primitive {

public:

    Vector3 center;
    float radius;

    // __device__ Sphere() {
    //     this->type = Primitive::SPHERE;
    //     this->center = Vector3();
    //     this->radius = 1.0;
    // }

    // __device__ Sphere(const Vector3& center, const float& radius) {
    //     this->type = Primitive::SPHERE;
    //     this->center = Vector3(center);
    //     this->radius = radius;
    // }

    __device__ float raycast(const Ray& ray) const {

        // Get coefficients for quadratic equation (ax^2 + bx + c = 0)
        float a = Vector3::dot(ray.direction, ray.direction);
        //dot3(d, d, &a);

        Vector3 diff = Vector3::subtract(ray.origin, this->center);
        //subtract3(o, ce, diff);

        float b = Vector3::dot(ray.direction, diff);
        //dot3(d, diff, &b);
        b *= 2;

        float c = Vector3::dot(diff, diff);
        // dot3(diff, diff, &c);
        c = c - (this->radius*this->radius);

        //free(diff);

        // Solve quadratic equation
        float radicand = b*b - (4*a*c);
        if (radicand < 0) {
            // Solutions are imaginary, ray never intersects sphere.
            return -1.0;
        }

        float root = sqrtf(radicand);

        // TODO: Clean this up to be more efficient in knowing which will be max t value.
        float t1 = (-b - root) / (2*a);
        float t2 = (-b + root) / (2*a);

        if (t1 > 0 && (t1 < t2 || t2 < 0)) {
            return t1;
        }
        else if (t2 > 0 && (t2 < t1 || t1 < 0)) {
            return t2;
        }
        else {
            return -1.0;
        }
    }

    __device__ Vector3 getNormal(const Ray& ray, float distance, const Vector3& hitPoint) const {
        Vector3 normal = Vector3::subtract(hitPoint, this->center);
        normal.normalize();
        return normal;
    }

    // // DEFAULT
    // __device__ float raycast(const Ray& ray, Vector3*& hitPoint) const {
    //     float t = this->raycast(ray);
    //     if (t >= 0) {
    //         *hitPoint = ray.getPoint(t);
    //     } else {
    //         hitPoint = NULL; // Ray did not hit anything, make hit point NULL
    //     }
    //     return t;
    // }

    // // DEFAULT
    // __device__ float raycast(const Ray& ray, Vector3*& hitPoint, Vector3*& normal) const {
    //     float t = this->raycast(ray, hitPoint);

    //     // Check if ray didn't hit primitive
    //     if (hitPoint == NULL) {
    //         normal = NULL;
    //         return t;
    //     }

    //     *normal = this->getNormal(*hitPoint);
    //     return t;
    // }


};


#endif // SPHERE_H