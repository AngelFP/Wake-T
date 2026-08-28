// cppimport
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-fopenmp', '-O3', '-ffast-math', '-march=native'] if OSError != "nt" else ['/openmp']
cfg['extra_link_args'] = ['-lgomp'] if OSError != "nt" else []
%>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <complex>
#include <vector>
#include <atomic>
#include <omp.h>
#include <thread>
#include <chrono>
#include <execution>

namespace py = pybind11;

template<class T>
struct FArray1D {
    T* ptr = nullptr;

    FArray1D () = default;

    FArray1D (py::array_t<T>& arr) {
        py::buffer_info arr_info = arr.request();
        if (arr_info.ndim != 1) {
            throw std::invalid_argument("Wrong number of dimensions for input array");
        }
        ptr = static_cast<T*>(arr_info.ptr);
    }

    T& operator()(int i) const {
        return ptr[i];
    }
};

template<class T>
FArray1D(py::array_t<T>& arr) -> FArray1D<T>;

template<class T>
struct FArray2D {
    T* ptr = nullptr;
    long long int stride = 0;

    FArray2D () = default;

    FArray2D (py::array_t<T>& arr) {
        py::buffer_info arr_info = arr.request();
        if (arr_info.ndim != 2) {
            throw std::invalid_argument("Wrong number of dimensions for input array");
        }
        ptr = static_cast<T*>(arr_info.ptr);
        stride = arr_info.strides[0] / arr_info.itemsize;
    }

    T& operator()(int i, int j) const {
        return ptr[i + j * stride];
    }
};

template<class T>
FArray2D(py::array_t<T>& arr) -> FArray2D<T>;

template<class T>
struct FArray3D {
    T* ptr = nullptr;
    long long int jstride = 0;
    long long int kstride = 0;

    FArray3D () = default;

    FArray3D (py::array_t<T>& arr) {
        py::buffer_info arr_info = arr.request();
        if (arr_info.ndim != 3) {
            throw std::invalid_argument("Wrong number of dimensions for input array");
        }
        ptr = static_cast<T*>(arr_info.ptr);
        kstride = arr_info.strides[0] / arr_info.itemsize;
        jstride = arr_info.strides[1] / arr_info.itemsize;
    }

    T& operator()(int i, int j, int k) const {
        return ptr[i + j * jstride + k * kstride];
    }
};

template<class T>
FArray3D(py::array_t<T>& arr) -> FArray3D<T>;

using cplx = std::complex<double>;


void TDMA(
    long long int n,
    const double* a,
    const cplx* b,
    const double* c,
    const cplx* d,
    cplx* p
)
{
    std::vector<cplx> w(n-1);
    std::vector<cplx> g(n);

    w[0] = c[0] / b[0];
    g[0] = d[0] / b[0];

    for (int i=1; i<n-1; ++i) {
        double a_im1 = a[i-1];
        cplx inv_coef = 1.0 / (b[i] - a_im1 * w[i - 1]);
        g[i] = (d[i] - a_im1 * g[i - 1]) * inv_coef;
        w[i] = c[i] * inv_coef;
    }

    g[n-1] = (d[n-1] - a[n-3] * g[n-2]) / (b[n-1] - a[n-3] * w[n-3]);

    p[n-1] = g[n-1];
    for (int i=n-1; i>0; --i) {
        p[i - 1] = g[i - 1] - w[i - 1] * p[i];
    }
}


void recv (
    std::vector<std::atomic<int>>& current_time_step,
    int step, int slice
)
{
    while (step != current_time_step[slice].load(std::memory_order_acquire)) {
        // __builtin_ia32_pause();
        // using namespace std::chrono_literals;
        // std::this_thread::sleep_for(1us);
    }
}

void send (
    std::vector<std::atomic<int>>& current_time_step,
    int step, int slice
)
{
    current_time_step[slice].store(step, std::memory_order_release);
}

void DoGridIonization (
    int num_ion_species,
    FArray3D<double>& ion_densities,
    FArray2D<double>& elec_density,
    FArray2D<double>& chi_arr,
    const cplx* a_arr_this,
    const cplx* a_arr_prev,
    FArray1D<int>& ion_start_index,
    FArray1D<int>& ion_atomic_number,
    FArray1D<double>& ion_mass,
    double omega0,
    FArray3D<double>& adk_prefactors,
    bool is_linear_pol,
    int j, int r_begin, int r_end,
    double d_zeta_inv
)
{
    constexpr double ct_m_e = 9.1093837139e-31;
    constexpr double ct_c = 299792458.;
    constexpr double ct_e = 1.602176634e-19;

    for (int i_s=0; i_s<num_ion_species; ++i_s) {
        double chi_factor_ion = ct_m_e / ion_mass(i_s);
        bool is_last_plasma = i_s + 1 == num_ion_species;
        int max_ion_lev = ion_atomic_number(i_s);
        int start_idx = ion_start_index(i_s);

        for (int k=r_begin; k<r_end; ++k) {

            if (i_s == 0) {
                chi_arr(k, j) = 0;
                elec_density(k, j) = elec_density(k, j + 1);
            }
            for (int ion_lev=0; ion_lev <= max_ion_lev; ++ion_lev) {
                ion_densities(k, j, start_idx + ion_lev)
                    = ion_densities(k, j + 1, start_idx + ion_lev);
            }

            cplx Et = cplx(0, 1) * a_arr_this[k] * omega0;
            Et += (a_arr_prev[k] - a_arr_this[k]) * ct_c * d_zeta_inv;
            double Ep = std::abs(Et);
            Ep *= ct_m_e * ct_c / ct_e;

            if (Ep <= 1e9) {
                continue;
            }

            double chi = 0;

            for (int ion_lev=0; ion_lev < max_ion_lev; ++ion_lev) {
                double p = 0;
                if (Ep > 1e-30) {
                    double w_dtau_dc = (
                        adk_prefactors(1, ion_lev, i_s)
                        * std::pow(Ep,adk_prefactors(0, ion_lev, i_s))
                        * std::exp(adk_prefactors(2, ion_lev, i_s) / Ep)
                    );

                    double w_dtau_ac = w_dtau_dc;
                    if (is_linear_pol) {
                        w_dtau_ac *= std::sqrt(Ep * adk_prefactors(3, ion_lev, i_s));
                    }

                    p = 1 - std::exp(-w_dtau_ac);
                }

                double old_weight = ion_densities(k, j, start_idx + ion_lev);
                double transferred_weight = old_weight * p;
                double new_weight = old_weight - transferred_weight;

                chi += new_weight * chi_factor_ion * ion_lev * ion_lev;

                ion_densities(k, j, start_idx + ion_lev) = new_weight;
                ion_densities(k, j, start_idx + ion_lev + 1) += transferred_weight;
                elec_density(k, j) += transferred_weight;
            }

            chi += ion_densities(k, j, start_idx + max_ion_lev)
                * chi_factor_ion * max_ion_lev * max_ion_lev;

            if (is_last_plasma) {
                chi += elec_density(k, j);
            }

            chi_arr(k, j) += chi;
        }
    }
}

void SolveOneSlice (
    int n, int j, int nz, int nr,
    std::vector<std::atomic<int>>& current_time_step,
    bool is_first_step,
    cplx C_minus, cplx C_plus,
    double inv_dzdt, double inv_dt,
    FArray2D<cplx>& a_arr,
    FArray2D<cplx>& a_old_arr,
    FArray2D<double>& chi_arr,
    std::vector<cplx>& rhs,
    std::vector<cplx>& a_new_jp1,
    std::vector<cplx>& a_new_jp2,
    std::vector<cplx>& d_main,
    const std::vector<double>& L_minus_over_2,
    const std::vector<double>& L_plus_over_2,
    int num_ion_species,
    FArray3D<double>& ion_densities,
    FArray3D<double>& initial_ion_densities,
    FArray2D<double>& elec_density,
    FArray2D<double>& initial_elec_density,
    FArray1D<int>& ion_start_index,
    FArray1D<int>& ion_atomic_number,
    FArray1D<double>& ion_mass,
    double omega0,
    FArray3D<double>& adk_prefactors,
    bool is_linear_pol,
    double d_zeta_inv
)
{
    if (j == nz-1) {
        recv(current_time_step, n, nz+1);
        recv(current_time_step, n, nz);
    }

    recv(current_time_step, n, j);

    if (is_first_step && n == 0) {
        for (int k=0; k<nr; ++k) {

            cplx rhs_k = (
                - ((C_minus - chi_arr(k, j) * 0.5) * a_arr(k, j))
                - (
                    4
                    * inv_dzdt
                    * (a_new_jp1[k] - a_arr(k, j + 1))
                )
                + (
                    1
                    * inv_dzdt
                    * (a_new_jp2[k] - a_arr(k, j + 2))
                )
            );

            if (k > 0) {
                rhs_k -= L_minus_over_2[k] * a_arr(k - 1, j);
            }
            if (k + 1 < nr) {
                rhs_k -= L_plus_over_2[k] * a_arr(k + 1, j);
            }
            rhs[k] = rhs_k;

            d_main[k] = C_plus - chi_arr(k, j) * 0.5;
        }
    } else {
        for (int k=0; k<nr; ++k) {

            cplx rhs_k = (
                -2 * inv_dt * inv_dt * a_arr(k, j)
                - ((C_minus - chi_arr(k, j) * 0.5) * a_old_arr(k, j))
                - (
                    2
                    * inv_dzdt
                    * (a_new_jp1[k] - a_old_arr(k, j + 1))
                )
                + (
                    0.5
                    * inv_dzdt
                    * (a_new_jp2[k] - a_old_arr(k, j + 2))
                )
            );

            if (k > 0) {
                rhs_k -= L_minus_over_2[k] * a_old_arr(k - 1, j);
            }
            if (k + 1 < nr) {
                rhs_k -= L_plus_over_2[k] * a_old_arr(k + 1, j);
            }
            rhs[k] = rhs_k;

            d_main[k] = C_plus - chi_arr(k, j) * 0.5;
        }
    }

    for (int k=0; k<nr; ++k) {
        a_old_arr(k, j+2) = a_arr(k, j+2);
        a_arr(k, j+2) = a_new_jp2[k];
    }

    send(current_time_step, n+1, j+2);

    std::swap(a_new_jp2, a_new_jp1);

    TDMA(nr, L_minus_over_2.data()+1, d_main.data(), L_plus_over_2.data(),
        rhs.data(), a_new_jp1.data());

    // calculate chi(k, j) using a_arr(k, j) = a_new_jp1 and and a_arr(k, j+1) = a_new_jp2

    if (j == nz - 1) {
        for (int k=0; k<nr; ++k) {
            elec_density(k, j+1) = initial_elec_density(k, n);
            for (int i_s=0; i_s<num_ion_species; ++i_s) {
                int max_ion_lev = ion_atomic_number(i_s);
                int start_idx = ion_start_index(i_s);
                for (int ion_lev=0; ion_lev <= max_ion_lev; ++ion_lev) {
                    ion_densities(k, j+1, start_idx + ion_lev) =
                        initial_ion_densities(k, n, start_idx + ion_lev);
                }
            }
        }
    }

    DoGridIonization(
        num_ion_species,
        ion_densities,
        elec_density,
        chi_arr,
        a_new_jp1.data(),
        a_new_jp2.data(),
        ion_start_index,
        ion_atomic_number,
        ion_mass,
        omega0,
        adk_prefactors,
        is_linear_pol,
        j, 0, nr, d_zeta_inv
    );

    if (j == 0) {
        for (int k=0; k<nr; ++k) {
            a_old_arr(k, 0) = a_arr(k, 0);
            a_old_arr(k, 1) = a_arr(k, 1);
            a_arr(k, 0) = a_new_jp1[k];
            a_arr(k, 1) = a_new_jp2[k];
        }

        send(current_time_step, n+1, 1);
        send(current_time_step, n+1, 0);
    }
}

void parallel_solver (
    py::array_t<cplx>& a,
    py::array_t<cplx>& a_old,
    py::array_t<double>& chi,
    double k0,
    double kp,
    double zmin,
    double zmax,
    long long int nz,
    double rmax,
    long long int nr,
    double dt,
    long long int nt,
    bool use_phase,
    bool is_first_step,
    int num_ion_species,
    py::array_t<double>& ion_densities_arr,
    py::array_t<double>& initial_ion_densities_arr,
    py::array_t<double>& elec_density_arr,
    py::array_t<double>& initial_elec_density_arr,
    py::array_t<int>& ion_start_index_arr,
    py::array_t<int>& ion_atomic_number_arr,
    py::array_t<double>& ion_mass_arr,
    double omega0,
    py::array_t<double>& adk_prefactors_arr,
    bool is_linear_pol,
    double d_zeta_inv,
    int num_threads
)
{
    FArray2D a_arr{a};
    FArray2D a_old_arr{a_old};
    FArray2D chi_arr{chi};
    FArray3D ion_densities{ion_densities_arr};
    FArray3D initial_ion_densities{initial_ion_densities_arr};
    FArray2D elec_density{elec_density_arr};
    FArray2D initial_elec_density{initial_elec_density_arr};
    FArray1D ion_start_index{ion_start_index_arr};
    FArray1D ion_atomic_number{ion_atomic_number_arr};
    FArray1D ion_mass{ion_mass_arr};
    FArray3D adk_prefactors{adk_prefactors_arr};

    py::gil_scoped_release release;

    omp_set_num_threads(num_threads);

    constexpr double ct_c = 299792458.;

    double dz = (zmax - zmin) * kp / (nz - 1);
    double dr = rmax * kp / nr;
    dt = dt * ct_c * kp;

    double inv_dt = 1 / dt;
    double inv_dr = 1 / dr;
    double inv_dz = 1 / dz;
    double inv_dzdt = inv_dt * inv_dz;
    double k0_over_kp = k0 / kp;

    std::vector<double> L_minus_over_2(nr);
    std::vector<double> L_plus_over_2(nr);

    for (int i=0; i<nr; ++i) {
        double L_base = 1.0 / (2.0 * (i + 0.5));
        L_minus_over_2[i] = (1.0 - L_base) * inv_dr*inv_dr* 0.5;
        L_plus_over_2[i] = (1.0 + L_base) * inv_dr*inv_dr * 0.5;
    }

    std::vector<std::atomic<int>> current_time_step (nz+2);
    for (auto& step : current_time_step) {
        step = 0;
    }

#pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        std::vector<cplx> rhs(nr);
        std::vector<cplx> a_new_jp1(nr);
        std::vector<cplx> a_new_jp2(nr);
        std::vector<cplx> d_main(nr);

        for (int n=ithread; n<nt; n += nthreads) {
            for (int k=0; k<nr; ++k) {
                a_new_jp1[k] = 0;
                a_new_jp2[k] = 0;
            }

            cplx C_minus = 0;
            cplx C_plus = 0;
            if (is_first_step && n == 0) {
                C_minus =
                    -2.0 * inv_dr*inv_dr * 0.5 - cplx(0, 2) * k0_over_kp * inv_dt
                    + 3 * inv_dzdt;
                C_plus =
                    -2.0 * inv_dr*inv_dr* 0.5 + cplx(0, 2) * k0_over_kp * inv_dt
                    - 3 * inv_dzdt;
            } else {
                C_minus =
                    -2.0 * inv_dr * inv_dr* 0.5 - cplx(0, 1) * k0_over_kp * inv_dt
                    + 1.5 * inv_dzdt - inv_dt*inv_dt;
                C_plus =
                    -2.0 * inv_dr * inv_dr * 0.5 + cplx(0, 1) * k0_over_kp * inv_dt
                    - 1.5 * inv_dzdt - inv_dt*inv_dt;
            }

            for (int j=nz-1; j>=0; --j) {
                SolveOneSlice (
                    n, j, nz, nr,
                    current_time_step,
                    is_first_step,
                    C_minus, C_plus,
                    inv_dzdt, inv_dt,
                    a_arr,
                    a_old_arr,
                    chi_arr,
                    rhs,
                    a_new_jp1,
                    a_new_jp2,
                    d_main,
                    L_minus_over_2,
                    L_plus_over_2,
                    num_ion_species,
                    ion_densities,
                    initial_ion_densities,
                    elec_density,
                    initial_elec_density,
                    ion_start_index,
                    ion_atomic_number,
                    ion_mass,
                    omega0,
                    adk_prefactors,
                    is_linear_pol,
                    d_zeta_inv
                );
            }
        }
    }
}

void calculate_grid_ionization (
    py::array_t<cplx>& a,
    py::array_t<double>& chi,
    long long int nz,
    long long int nr,
    int num_ion_species,
    py::array_t<double>& ion_densities_arr,
    py::array_t<double>& elec_density_arr,
    py::array_t<int>& ion_start_index_arr,
    py::array_t<int>& ion_atomic_number_arr,
    py::array_t<double>& ion_mass_arr,
    double omega0,
    py::array_t<double>& adk_prefactors_arr,
    bool is_linear_pol,
    double d_zeta_inv,
    int num_threads
)
{
    FArray2D a_arr{a};
    FArray2D chi_arr{chi};
    FArray3D ion_densities{ion_densities_arr};
    FArray2D elec_density{elec_density_arr};
    FArray1D ion_start_index{ion_start_index_arr};
    FArray1D ion_atomic_number{ion_atomic_number_arr};
    FArray1D ion_mass{ion_mass_arr};
    FArray3D adk_prefactors{adk_prefactors_arr};

    py::gil_scoped_release release;

    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        const int r_begin = (std::size_t(ithread) * nr) / nthreads;
        const int r_end = (std::size_t(ithread+1) * nr) / nthreads;

        for (int j=nz-1; j>=0; --j) {
            DoGridIonization(
                num_ion_species,
                ion_densities,
                elec_density,
                chi_arr,
                &a_arr(0, j),
                &a_arr(0, j+1),
                ion_start_index,
                ion_atomic_number,
                ion_mass,
                omega0,
                adk_prefactors,
                is_linear_pol,
                j, r_begin, r_end, d_zeta_inv
            );
        }
    }
}

PYBIND11_MODULE(parallel_solver, m) {
    m.def("parallel_solver", &parallel_solver, "doc");
    m.def("calculate_grid_ionization", &calculate_grid_ionization, "doc");
}
