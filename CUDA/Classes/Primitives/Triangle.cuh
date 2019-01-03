#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <math.h>
#include "Vector3.cuh"
#include "Primitive.cuh"


class Triangle: public Primitive {

public:

    Vector3 v0, v1, v2; // Vertex coordinates
    Vector3 normal;     // Face Normal
    Vector3 d0, d1, d2; // Vertex data

    __device__ float raycast(const Ray& ray) const {
        return this->raycast(ray, NULL);
    }

    __device__ float raycast(const Ray& ray, Vector3 * uvw) const {

        // Vector3 * o = &ray.origin;
        //Vector3 * d = &ray.direction;
        //Vector3 * n = &this->normal;



        // float * vertices = triangle + 1;
        // float * a = vertices;
        // float * b = a + 3;
        // float * c = b + 3;

        // Vector3 * a = this->v0;
        // Vector3 * b = this->v1;
        // Vector3 * c = this->v2;
        
        // float * buffer = (float*)malloc(6 * 3 * sizeof(float));
        // float * diff = buffer;
        // float * coords = buffer + 3;
        // float * v0 = buffer + 6;
        // float * v1 = buffer + 9;
        // float * v2 = buffer + 12;
        // float * hit = buffer + 15;

        // Allocate
        Vector3 diff = Vector3();
        Vector3 coords = Vector3();
        Vector3 vec0 = Vector3();
        Vector3 vec1 = Vector3();
        Vector3 vec2 = Vector3();
        Vector3 hit = Vector3();

        // Check if ray intersects the triangle's plane
        int intersectsPlane = 0;
        // float denom;
        // dot3(n, d, &denom);

        float denom = Vector3::dot(this->normal, ray.direction);
        

        float tVal;
        if (denom != 0.0) {

            //subtract3(a, o, diff);
            diff = Vector3::subtract(this->v0, ray.origin);

            // float numerator;
            // dot3(n, diff, &numerator);

            float numerator = Vector3::dot(this->normal, diff);

            tVal = numerator/denom;
            
            if (tVal > 0) {
                intersectsPlane = 1;
            }
        }

        if (!intersectsPlane) {
            return -1.0;
        }

        // Calculate barycentric coordinates to determine if the ray interesects the plane within the triangle
        
        // hit[0] = o[0] + d[0] * tVal;
        // hit[1] = o[1] + d[1] * tVal;
        // hit[2] = o[2] + d[2] * tVal;

        hit = Vector3::add(ray.origin, Vector3::multiply(ray.direction, tVal));

        // subtract3(b, a, v0);
        // subtract3(c, a, v1);
        // subtract3(hit, a, v2);

        vec0 = Vector3::subtract(this->v1, this->v0);
        vec1 = Vector3::subtract(this->v2, this->v0);
        vec2 = Vector3::subtract(hit, this->v0);

        // float d00;
        // float d01;
        // float d11;
        // float d20;
        // float d21;

        // dot3(v0, v0, &d00);
        // dot3(v0, v1, &d01);
        // dot3(v1, v1, &d11);
        // dot3(v2, v0, &d20);
        // dot3(v2, v1, &d21);

        float d00 = Vector3::dot(vec0,vec0);
        float d01 = Vector3::dot(vec0,vec1);
        float d11 = Vector3::dot(vec1,vec1);
        float d20 = Vector3::dot(vec2,vec0);
        float d21 = Vector3::dot(vec2,vec1);

        
        denom = d00*d11 - d01*d01;
        
        coords.y = (d11*d20 - d01*d21)/denom; // v
        coords.z = (d00*d21 - d01*d20)/denom; // w
        coords.x = 1.0 - coords.y - coords.z; // u

        if (coords.x < 0 || coords.y < 0 ||coords.z < 0) {
            return -1.0; // Hit point falls outside of triangle
        }

        // Ray intersects triangle!
        
        if (tVal > 0) { // TODO: MOVE THIS CHECK TO END OF PLANE INTERSECTION , THIS CAN BE CHECKED THEN
            if (uvw != NULL) {
                *uvw = Vector3(coords);
            }
            
            return tVal;
        } else {
            return -1.0; // Intersection behind ray
        }
    }

    __device__ Vector3 getNormal(const Ray& ray, float distance, const Vector3& hitPoint) const {
        if (d0.x == 0.0 && d0.y == 0.0 && d0.z == 0.0) {
            // No vertex data, just return face normal
            return this->normal;
        }

        // There is normal data, interpolate it
        Vector3 uvw = Vector3();
        this->raycast(ray, &uvw); // We know it will intersect at this point

        Vector3 n = this->interpolateData(uvw);
        n.normalize();

        return n;
    }

    __device__ Vector3 interpolateData(Vector3 uvw) const {
        Vector3 wd0 = Vector3::multiply(this->d0, uvw.x);
        Vector3 wd1 = Vector3::multiply(this->d1, uvw.y);
        Vector3 wd2 = Vector3::multiply(this->d2, uvw.z);

        Vector3 average = Vector3(0,0,0);
        average.add(wd0);
        average.add(wd1);
        average.add(wd2);

        average.divide(3.0);

        return average;
    }


};


#endif // TRIANGLE_H