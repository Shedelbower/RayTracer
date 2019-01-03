#ifndef MATERIAL_H
#define MATERIAL_H

#include "RGBColor.cuh"

class Material {

public:
    float type; // Unused in this class, needed for offset
    float ambientWeight, diffuseWeight, specularWeight;
    RGBColor ambientColor, diffuseColor, specularColor;
    float shininess;
    float mirror, transparent;

    // CONSTRUCTORS

    // Defaul Constructor
    __device__ Material();

    // Main Constructor
    __device__ Material(float ambientWeight, float diffuseWeight, float specularWeight, 
                RGBColor ambientColor, RGBColor diffuseColor, RGBColor specularColor,
                int shininess, float mirror, float transparent);
    
    // Deconstructor
    __device__ ~Material();

    // MEMBER FUNCTIONS

    __device__ bool isPerfectMirror();

    __device__ bool isTransparent();

};


#include "Material.cu"


#endif // MATERIAL_H