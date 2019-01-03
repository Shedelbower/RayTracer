#ifndef LIGHT_H
#define LIGHT_H


#include "RGBColor.cuh"
#include "Vector3.cuh"
#include "BVHNode.cuh"
#include "Primitive.cuh"
#include "Raycast.cuh"

class Light {

public:

    enum Type {
        DIRECTIONAL = 100,
        POINT = 101
    };

    float typef;
    float intensity;
    RGBColor color;
    Vector3 positionOrDirection; // Position for Point Light, Direction for Directional Light
    float attenuationConstant, attenuationLinear, attenuationExponential; // Point Light Only!

    // Returns Vector3 of ambient, diffuse, and specular intensities from this light
    // View vec should be normalized and go from surface towards eye point
    __device__ Vector3 getDirectIrradiance(const Vector3& point, const Vector3& normal, const Vector3& viewVec, Material * material, BVHNode * bvh, Primitive * primitives, bool enableShadows) const {
        
        Vector3 lightVec = this->getLightVec(point);
        //lightVec.reverse(); // Move later?

        if (enableShadows && this->pointInShadow(point, lightVec, bvh, primitives)) {
            return Vector3(0.0, 0.0, 0.0); // Point has no direct path to light, so no contribution is provided
        }

        float ambient = 0.0; // TODO: Should I do something with this?
        float diffuse = this->getPhongDiffuse(normal, lightVec);
        float specular = this->getPhongSpecular(normal, lightVec, viewVec, material->shininess);

        float attenuation = this->getAttenuation(point);

        Vector3 irradiance = Vector3(ambient, diffuse, specular);
        irradiance.multiply(intensity/(attenuation + 0.0001));

        return irradiance;
    }

    __device__ bool pointInShadow(const Vector3& point, const Vector3& lightVec, BVHNode * bvh, Primitive * primitives) const {
        // Do shadow ray calculations

        Ray shadowRay = Ray(point, lightVec);

        float distance = -1.0;
        Primitive * primitive = NULL;

        Raycast::raycast(&shadowRay, bvh, primitives,
                &distance, &primitive, NULL, NULL
        );

        if (distance < 0) {
            return false;
        }

        Light::Type type = (Light::Type) this->typef;
        if (type == Light::POINT) {
            Vector3 surfaceToLight = Vector3::subtract(point, this->positionOrDirection);
            float lightDist = surfaceToLight.magnitude();
            return lightDist > distance;
        } else {
            return true; // Distance is greater than 0, meaning some primitive was hit
        }
    }

    __device__ Vector3 getLightVec(const Vector3& point) const {
        Light::Type type = (Light::Type) this->typef;
        if (type == Light::POINT) {
            Vector3 surfaceToLight = Vector3::subtract(this->positionOrDirection, point);
            surfaceToLight.normalize();
            return surfaceToLight;
        } else {
            Vector3 lightVec = this->positionOrDirection;
            lightVec.reverse();
            return lightVec;
        }
    }

    __device__ float getAttenuation(const Vector3& point) const {
        Light::Type type = (Light::Type) this->typef;
        if (type == Light::POINT) {
            Vector3 surfaceToLight = Vector3::subtract(point, this->positionOrDirection);
            float distance = surfaceToLight.magnitude();
            return this->attenuationConstant + this->attenuationLinear * distance + this->attenuationExponential * distance * distance;
        } else {
            return 1.0;
        }
    }


private:


    // lightVec is the normalized vector from the surface point to the light
    __device__ float getPhongDiffuse(const Vector3& normal, const Vector3& lightVec) const {
        float dot = Vector3::dot(normal, lightVec);
        if (dot < 0) {
            dot = 0;
        }
        return dot;
    }

    // lightVec is the normalized vector from the surface point to the eye/viewer
    __device__ float getPhongSpecular(const Vector3& normal, const Vector3& lightVec, const Vector3& viewVec, int shininess) const {

        Vector3 halfwayVec = Vector3::add(lightVec, viewVec);
        halfwayVec.normalize();

        float dot = Vector3::dot(normal, halfwayVec);

        float specular = 1.0;
        for (int i = 0; i < shininess; i++) {
            specular *= dot;
        }

        return specular;
    }

};



#endif // LIGHT_H