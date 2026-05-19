#ifndef _XY_H_
#define _XY_H_

#include <cstring>
#include <random>
#include <vector>
#include <cmath>


class XY
{
public:
    XY(float t, int L) : L(L), site_number(L * L), t(t)
    {
        num_components = 2;
        spin = new float[num_components * site_number];
        to_visit = new int[num_components * site_number];
        in_cluster = new bool[site_number];
        e = new float[flush_length];
        m = new float[flush_length];
        h = new float[flush_length];

        // Initialize spins to point along x-axis
        for (int i = 0; i < site_number; i++)
        {
            spin[num_components * i] = 1.0f;
            spin[num_components * i + 1] = 0.0f;
        }
    }

    float get_t() const { return t; };
    int get_L() const { return L; };
    float *get_spin() const { return spin; };
    float *get_e() const { return e; };
    float *get_m() const { return m; };
    float *get_h() const { return h; };
    int get_flush_length() const {return flush_length; };

    void compute_observables(int flush_length);
    void set_spin(const float *new_spin)
    {
        std::memcpy(spin, new_spin, sizeof(float) * 2 * site_number);
    }

    void run(int spacing);

    ~XY()
    {
        delete[] spin;
        delete[] in_cluster;
        delete[] e;
        delete[] m;
        delete[] h;
        delete[] to_visit;
    }

private:
    const float t;
    const int L, site_number, flush_length=1024;
    int num_components = 2;
    float *spin;
    float *e, *m, *h;
    bool *in_cluster;
    int *to_visit;
    void update(int x0, int y0);
    int idx(int x, int y) const { return num_components * (x * L + y); }
};

#endif