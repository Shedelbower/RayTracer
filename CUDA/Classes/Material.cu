#include "RGBColor.cuh"

// Defaul Constructor
__device__ Material::Material() {
    this->ambientWeight = 0.0f;
    this->diffuseWeight = 0.0f;
    this->specularWeight = 0.0f;

    this->ambientColor = RGBColor();
    this->diffuseColor = RGBColor();
    this->specularColor = RGBColor();

    this->shininess = 1;

    this->mirror = 0.0f;
    this->transparent = 0.0f;
}

// Main Constructor
__device__ Material::Material(float ambientWeight, float diffuseWeight, float specularWeight, 
            RGBColor ambientColor, RGBColor diffuseColor, RGBColor specularColor,
            int shininess, float mirror, float transparent):
            ambientWeight(ambientWeight), diffuseWeight(diffuseWeight), specularWeight(specularWeight), 
            ambientColor(ambientColor), diffuseColor(diffuseColor), specularColor(specularColor),
            shininess(shininess), mirror(mirror), transparent(transparent) {}

// Deconstructor
__device__ Material::~Material() {
    // Do nothing
}


// MEMBER FUNCTIONS

__device__ bool Material::isPerfectMirror() {
    return this->mirror > 0;
}

__device__ bool Material::isTransparent() {
    return this->transparent > 0;
}
