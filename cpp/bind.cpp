#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "xy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(XY, m) {
    m.doc() = "XY model simulation using Wolff cluster algorithm";
    py::class_<XY>(m, "XY")
        .def(py::init<float, int>(), py::arg("t"), py::arg("L"),
             "Constructor: XY(temperature, lattice_size)")

        // Return spin array as a 1D numpy view (directly maps internal memory)
        .def("get_spin", [](XY& self) {
            float* data = self.get_spin();
            size_t size = 2 * self.get_L() * self.get_L();
            return py::array_t<float>(
                { size },                 // shape
                { sizeof(float) },        // strides (contiguous)
                data,                     // data pointer
                py::cast(self)            // owner object, keeps view alive
            );
        }, "Return spin array as 1D numpy array of shape (2*L*L,)")

        .def("get_L", &XY::get_L, "Return lattice size L")

        .def("get_e", [](XY& self) {
            float* data = self.get_e();
            size_t size = self.get_flush_length();
            return py::array_t<float>(
                { size }, { sizeof(float) }, data, py::cast(self)
            );
        }, "Return energy array of length flush_length")

        .def("get_m", [](XY& self) {
            float* data = self.get_m();
            size_t size = self.get_flush_length();
            return py::array_t<float>(
                { size }, { sizeof(float) }, data, py::cast(self)
            );
        }, "Return magnetization array of length flush_length")

        .def("get_h", [](XY& self) {
            float* data = self.get_h();
            size_t size = self.get_flush_length();
            return py::array_t<float>(
                { size }, { sizeof(float) }, data, py::cast(self)
            );
        }, "Return raw sin(dtheta) array of length flush_length")

        .def("set_spin", [](XY& self, py::array_t<float> arr) {
            py::buffer_info buf = arr.request();
            size_t expected = 2 * self.get_L() * self.get_L();
            if (buf.size != expected)
                throw std::runtime_error("set_spin: array size must be 2*L*L");
            if (buf.ndim != 1)
                throw std::runtime_error("set_spin: array must be 1-dimensional");
            if (buf.strides[0] != sizeof(float))
                throw std::runtime_error("set_spin: array must be C-contiguous");
            self.set_spin(static_cast<float*>(buf.ptr));
        }, py::arg("arr"), "Copy data from a 1D numpy array into spin array")

        .def("run", &XY::run, py::arg("spacing"),
             "Perform one flush: run 'spacing' cluster updates, then record observables")

        .def("helicity_modulus", [](XY& self) {
            float* e_data = self.get_e();
            float* h_data = self.get_h();
            int n = self.get_flush_length();
            int L = self.get_L();
            float t = self.get_t();
            float sum_e = 0.0f;
            float sum_h2 = 0.0f;
            for (int i = 0; i < n; i++) {
                sum_e += e_data[i];
                sum_h2 += h_data[i] * h_data[i];
            }
            float mean_e = sum_e / static_cast<float>(n);
            float mean_h2 = sum_h2 / static_cast<float>(n);
            // Upsilon = <cos(dtheta_x)> - (beta/N) * <(Sigma sin)^2>
            //          = -<e>/2 - (L^2 / T) * <h^2>
            return -mean_e / 2.0f - (static_cast<float>(L * L) / t) * mean_h2;
        }, "Compute helicity modulus from the current flush buffer: "
           "Upsilon = -<e>/2 - (L^2/T)*<h^2>");
}