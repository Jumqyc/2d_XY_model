#include "xy.hpp"
#include <iostream>

void XY::update(int x0, int y0)
{
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    float theta = unif(rng) * 2.0f * M_PI;
    float axis_x = cos(theta);
    float axis_y = sin(theta);
    std::fill(in_cluster, in_cluster + site_number, false);

    to_visit[0] = x0;
    to_visit[1] = y0;
    int ptr = 1; // Points to the last element in to_visit


    while (ptr > 0)
    {
        int y = to_visit[ptr];
        ptr--;
        int x = to_visit[ptr];
        ptr--;


        int ind = idx(x, y);

        in_cluster[ind / num_components] = true;
        float proj = axis_x * spin[ind] + axis_y * spin[ind + 1];
        // Flip the spin with respect to the chosen axis
        spin[ind] -= 2.0f * proj * axis_x;
        spin[ind + 1] -= 2.0f * proj * axis_y;

        // Check neighbors
        for (auto [nx, ny] : {std::pair{(x + 1) % L, y},
                              std::pair{(x + L - 1) % L, y},
                              std::pair{x, (y + 1) % L},
                              std::pair{x, (y + L - 1) % L}})
        {
            int ind_n = idx(nx, ny);
            if (in_cluster[ind_n / num_components])
                continue;
            float prod = (axis_x * spin[ind_n] + axis_y * spin[ind_n + 1]) * proj;
            if (prod < 0)
                continue;
            if (unif(rng) > std::exp(-2.0f * (prod / t)))
            {
                in_cluster[ind_n / num_components] = true;
                ptr++;
                to_visit[ptr] = nx;
                ptr++;
                to_visit[ptr] = ny;
            }
        }
    }
}

void XY::run(int spacing)
{
    static std::random_device dev;
    static std::mt19937 rng(dev());
    std::uniform_int_distribution<int> uniint(0, L - 1);
    for (int flush_idx = 0; flush_idx < flush_length; flush_idx++)
    {
        for (int i = 0; i < spacing; i++)
        {
            int x0 = uniint(rng);
            int y0 = uniint(rng);
            update(x0, y0);
        }
        compute_observables(flush_idx);
    }
}

void XY::compute_observables(int flush_idx)
{
    float mx = 0.0f;
    float my = 0.0f;
    float e_local = 0.0f;
    float h_local = 0.0f;
    for (unsigned int i = 0; i < L; i++)
    {
        for (unsigned int j = 0; j < L; j++)
        {
            unsigned int this_idx = idx(i, j);
            float sx = spin[this_idx];
            float sy = spin[this_idx + 1];
            mx += sx;
            my += sy;
            // Interaction with right neighbor (periodic)
            unsigned int right_idx = idx(i, (j + 1) % L);
            e_local -= (sx * spin[right_idx] + sy * spin[right_idx + 1]);
            // Interaction with bottom neighbor (periodic)
            unsigned int bottom_idx = idx((i + 1) % L, j);
            e_local -= (sx * spin[bottom_idx] + sy * spin[bottom_idx + 1]);
            // Helicity calculation
            h_local += (sx * spin[right_idx + 1] - sy * spin[right_idx]);
        }
    }
    m[flush_idx] = std::sqrt(mx * mx + my * my) / static_cast<float>(site_number);
    e[flush_idx] = e_local / static_cast<float>(site_number);
    h[flush_idx] = h_local / static_cast<float>(site_number);
}