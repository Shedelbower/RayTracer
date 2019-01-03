#ifndef RGBCOLOR_H
#define RGBCOLOR_H

class RGBColor {

public:
    float red, green, blue;

    // CONSTRUCTORS

    // Default Constructor
    __device__ RGBColor();

    // Main Constructor
    __device__ RGBColor(float red, float green, float blue);

    // Copy Constructor
    __device__ RGBColor(const RGBColor& other);

    // MEMBER FUNCTIONS

    __device__ void add(const RGBColor& other);

    __device__ void multiply(const float& value);


    // STATIC
    __device__ static RGBColor multiply(const RGBColor& color, const float& value);


};

#include "RGBColor.cu"

#endif // RGBCOLOR_H