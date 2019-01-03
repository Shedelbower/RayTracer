#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "Vector3.cuh"
#include "Ray.cuh"
#include "Material.cuh"


class Sphere;

class Primitive {

public:

    enum Type {
        NONE = -1,
        SPHERE = 0,
        PLANE = 1,
        TRIANGLE = 2
    };

    float typef;
    float materialIndex;

    __device__ float raycast(const Ray& ray) const;

    __device__ Vector3 getHitPoint(const Ray& ray, float distance) const {
        Vector3 hitPoint = Vector3::multiply(ray.direction, distance);
        hitPoint.add(ray.origin);
        return hitPoint;
    }

    __device__ Vector3 getNormal(const Ray& ray, float distance, const Vector3& hitPoint) const;

    __device__ Material * getMaterial(Material * materials) const {
        int index = (int) materialIndex;
        return materials + index;
    }

    __device__ Primitive::Type getType() const {
        return (Primitive::Type)(int)this->typef;
    }

    /*
    __device__ virtual Vector3 getNormal(const Vector3& point) const = 0;

    __device__ virtual float raycast(const Ray& ray) const {
        return -1.0;
    }

    
    __device__ virtual float raycast(const Ray& ray, Vector3*& hitPoint) const {
        float t = this->raycast(ray);
        if (t >= 0) {
            *hitPoint = ray.getPoint(t);
        } else {
            hitPoint = NULL; // Ray did not hit anything, make hit point NULL
        }
        return t;
    }

    __device__ virtual float raycast(const Ray& ray, Vector3*& hitPoint, Vector3*& normal) const {
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

// Include all primitives here
#include "Sphere.cuh"
#include "Triangle.cuh"

__device__ float Primitive::raycast(const Ray& ray) const {
    Primitive::Type type = this->getType();
    if (type == Primitive::SPHERE) {
        Sphere * sphere = (Sphere*) this;
        return sphere->raycast(ray);
    } else if (type == Primitive::PLANE) {
        //TODO
        return -1.0;
    } else if (type == Primitive::TRIANGLE) {
        Triangle * triangle = (Triangle*) this;
        return triangle->raycast(ray);
    }
    return (float)(int)type;
}

__device__ Vector3 Primitive::getNormal(const Ray& ray, float distance, const Vector3& hitPoint) const {
    Primitive::Type type = this->getType();
    if (type == Primitive::SPHERE) {
        Sphere * sphere = (Sphere*) this;
        return sphere->getNormal(ray, distance, hitPoint);
    } else if (type == Primitive::PLANE) {
        //TODO
    } else if (type == Primitive::TRIANGLE) {
        Triangle * triangle = (Triangle*) this;
        return triangle->getNormal(ray, distance, hitPoint);
    }
    return Vector3(-1.0, -1.0, -1.0);
}


#endif // PRIMITIVE_H