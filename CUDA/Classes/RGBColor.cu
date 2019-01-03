
#include "RGBColor.cuh"


// Default Constructor
__device__ RGBColor::RGBColor() {
    this->red = 0.0;
    this->green = 0.0;
    this->blue = 0.0;
}

// Main Constructor
__device__ RGBColor::RGBColor(float red, float green, float blue) : red(red), green(green), blue(blue) {}

// Copy Constructor
__device__ RGBColor::RGBColor(const RGBColor& other) : red(other.red), green(other.green), blue(other.blue) {}

__device__ void RGBColor::add(const RGBColor& other) {
    this->red += other.red;
    this->green += other.green;
    this->blue += other.blue;
}

__device__ void RGBColor::multiply(const float& value) {
    this->red *= value;
    this->green *= value;
    this->blue *= value;
}


__device__ RGBColor RGBColor::multiply(const RGBColor& color, const float& value) {
    RGBColor result = RGBColor(
        color.red * value,
        color.green * value,
        color.blue * value
    );
    return result;
}

