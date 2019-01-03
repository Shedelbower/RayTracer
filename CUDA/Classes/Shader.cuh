#ifndef SHADER_H
#define SHADER_H

#include <math.h>
#include "Vector3.cuh"
#include "Primitive.cuh"
#include "BVHNode.cuh"
#include "Light.cuh"
#include "Material.cuh"
#include "Ray.cuh"
#include "Image.cuh"

class Shader {

public:

    __device__ static Vector3 getDirectIrradiance(const Vector3& hitPoint, const Vector3& normal, const Ray& incomingRay, Material * material, BVHNode * bvh, Primitive * primitives, bool enableShadows, Light * lights, int lightCount) {

        Vector3 irradiance = Vector3(0.0, 0.0, 0.0);
        Vector3 viewVec = Vector3(incomingRay.direction);
        viewVec.reverse();
        Vector3 point = Vector3(hitPoint);
        float pointOffset = 0.001;
        point.add(Vector3::multiply(normal, pointOffset)); // Offset the point from the surface in the direction of the normal to avoid hitting same surface

        for (int lightIndex = 0; lightIndex < lightCount; lightIndex++) {
            Light * light = lights + lightIndex;
            irradiance.add(light->getDirectIrradiance(point, normal, viewVec, material, bvh, primitives, enableShadows));
        }

        return irradiance;
    }

    __device__ static RGBColor getColor(Vector3 * hitPoint, Vector3 * normal, const Ray& incomingRay, Material * material, BVHNode * bvh, Primitive * primitives, Material * materials, RGBColor backgroundColor, Image * skybox, bool enableSkybox, bool enableShadows, Light * lights, int lightCount) {
        if (material == NULL) {
            // Ray didn't hit anything
            if (enableSkybox) {
                return skybox->sample(incomingRay.direction);
            } else {
                return backgroundColor;
            }
        }
        
        RGBColor color = RGBColor();

        Vector3 irradiance =  Shader::getDirectIrradiance(
            *hitPoint, *normal, incomingRay, material, bvh, primitives, enableShadows, lights, lightCount
            );

        RGBColor ambient = RGBColor::multiply(material->ambientColor, material->ambientWeight);
        color.add(ambient);

        RGBColor diffuse = RGBColor::multiply(material->diffuseColor, irradiance.y);
        diffuse.multiply(material->diffuseWeight);
        color.add(diffuse);

        RGBColor specular = RGBColor::multiply(material->specularColor, irradiance.z);
        specular.multiply(material->specularWeight);
        color.add(specular);

        
        return color;
    }

};



#endif // SHADER_H