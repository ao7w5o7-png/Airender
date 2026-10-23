// The spectral Earth atmosphere behind BuildSkyTable (docs/architecture.md §8.9).
//
// Medium and constants follow Bruneton's 2017 reference implementation of precomputed atmospheric
// scattering: Rayleigh scattering 1.24062e-6 / lambda^4 per metre (lambda in micrometres) with an
// 8 km scale height; aerosols of a given optical depth (that reference uses 0.005, pristine air; 0.05
// by default here, a clear day) over a 1.2 km scale height, with a single-scattering albedo of 0.9
// and a Cornette-Shanks phase function with g = 0.8; 300
// Dobson units of ozone in a layer from 10 to 40 km peaking at 25 km; the measured solar irradiance
// outside the atmosphere and ozone absorption cross sections it tabulates. The light transport
// follows Hillaire 2020: a transmittance table, a multiple-scattering table from his isotropic
// approximation (all orders as a geometric series of the second), and a ray march per sky
// direction. Everything is computed for 16 wavelength bins of 25 nm (380-780 nm) and converted to
// linear sRGB through the CIE 1931 colour matching functions (the analytic fit of Wyman, Sloan and
// Shirley 2013), so the colours come from spectra rather than from three RGB "wavelengths". The same
// ray march, stopped at a series of distances, gives the aerial perspective tables.

#include "air/renderer/Sky.h"

#include "ParallelFor.h"

#include "air/core/Assert.h"
#include "air/renderer/Transfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace air::renderer
{
namespace
{

using detail::ParallelFor;
using hlsl::float3;

constexpr double kPi = std::numbers::pi;
constexpr int kBins = 16;
constexpr double kFirstNm = 380.0;
constexpr double kBinNm = 25.0;

using Spectrum = std::array<float, kBins>;

// --- Measured data, 360-830 nm in 10 nm steps --------------------------------------------------

constexpr int kTableCount = 48;
constexpr double kTableFirstNm = 360.0;
// Solar spectral irradiance outside the atmosphere, W m^-2 nm^-1.
constexpr double kSolarIrradiance[kTableCount] = {
    1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887,  1.61253, 1.91198, 2.03474, 2.02042, 2.02212,
    1.93377, 1.95809, 1.91686, 1.8298,  1.8685,  1.8931,  1.85149, 1.8504,  1.8341,  1.8345,  1.8147,  1.78158,
    1.7533,  1.6965,  1.68194, 1.64654, 1.6048,  1.52143, 1.55622, 1.5113,  1.474,   1.4482,  1.41018, 1.36775,
    1.34188, 1.31429, 1.28303, 1.26758, 1.2367,  1.2082,  1.18737, 1.14683, 1.12362, 1.1058,  1.07124, 1.04992};
// Ozone absorption cross section, m^2 per molecule.
constexpr double kOzoneCrossSection[kTableCount] = {
    1.18e-27,  2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,  8.451e-27, 1.582e-26, 2.316e-26,
    3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26, 1.48e-25,  1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25,
    4.266e-25, 4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25,  3.215e-25, 2.662e-25, 2.238e-25,
    1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26, 4.15e-26,  4.228e-26, 3.237e-26,
    2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27};

double TableAt(const double (&table)[kTableCount], double nm)
{
    const double x = (nm - kTableFirstNm) / 10.0;
    const int i = std::clamp(int(std::floor(x)), 0, kTableCount - 2);
    const double t = std::clamp(x - double(i), 0.0, 1.0);
    return table[i] + (table[i + 1] - table[i]) * t;
}

// Average of f over a wavelength bin, sampled every nanometre.
template <typename F> double BinAverage(int bin, F&& f)
{
    double sum = 0.0;
    for (int i = 0; i < int(kBinNm); ++i)
    {
        sum += f(kFirstNm + double(bin) * kBinNm + double(i) + 0.5);
    }
    return sum / kBinNm;
}

// CIE 1931 2-degree colour matching functions, multi-lobe fit (Wyman, Sloan and Shirley 2013).
double Lobe(double nm, double mu, double below, double above)
{
    const double t = (nm - mu) / (nm < mu ? below : above);
    return std::exp(-0.5 * t * t);
}

std::array<double, 3> Cmf(double nm)
{
    return {1.056 * Lobe(nm, 599.8, 37.9, 31.0) + 0.362 * Lobe(nm, 442.0, 16.0, 26.7) -
                0.065 * Lobe(nm, 501.1, 20.4, 26.2),
            0.821 * Lobe(nm, 568.8, 46.9, 40.5) + 0.286 * Lobe(nm, 530.9, 16.3, 31.1),
            1.217 * Lobe(nm, 437.0, 11.8, 36.0) + 0.681 * Lobe(nm, 459.0, 26.0, 13.8)};
}

// --- The atmosphere ------------------------------------------------------------------------------

struct Atmosphere
{
    double bottom = 6360e3; // planet radius, metres
    double top = 6460e3;    // 100 km of atmosphere
    double viewRadius = 6360e3 + 1.0;
    Spectrum rayleigh{};      // scattering at sea level, per metre
    Spectrum mieScattering{}; // at sea level, per metre
    Spectrum mieExtinction{};
    Spectrum ozone{}; // absorption at the layer's peak, per metre
    float groundAlbedo = 0.1f;
    Spectrum solar{};                               // irradiance outside the atmosphere, W m^-2 nm^-1
    std::array<std::array<double, 3>, kBins> cmf{}; // integral of each colour matching function over each bin
};

constexpr double kRayleighHeight = 8000.0;
constexpr double kMieHeight = 1200.0;
constexpr double kMieG = 0.8;

Atmosphere MakeAtmosphere(const Sky& sky)
{
    Atmosphere a;
    a.viewRadius = a.bottom + std::max(double(sky.altitude), 1.0);
    a.groundAlbedo = sky.groundAlbedo;
    // Extinction at sea level that integrates to the given optical depth over the scale height.
    const double mie = double(sky.aerosolDepth) / kMieHeight;
    const double ozoneDensity = 300.0 * 2.687e20 / 15000.0; // molecules per m^3 at the peak
    for (int b = 0; b < kBins; ++b)
    {
        a.rayleigh[b] = float(BinAverage(b, [](double nm) { return 1.24062e-6 / std::pow(nm * 1e-3, 4.0); }));
        a.mieExtinction[b] = float(mie);
        a.mieScattering[b] = float(mie * 0.9);
        a.ozone[b] = float(ozoneDensity * BinAverage(b, [](double nm) { return TableAt(kOzoneCrossSection, nm); }));
        a.solar[b] = float(BinAverage(b, [](double nm) { return TableAt(kSolarIrradiance, nm); }));
        for (int c = 0; c < 3; ++c)
        {
            a.cmf[b][c] = kBinNm * BinAverage(b, [c](double nm) { return Cmf(nm)[c]; });
        }
    }
    return a;
}

struct Densities
{
    float rayleigh;
    float mie;
    float ozone;
};

Densities DensitiesAt(double altitude)
{
    altitude = std::max(altitude, 0.0);
    return {float(std::exp(-altitude / kRayleighHeight)), float(std::exp(-altitude / kMieHeight)),
            float(std::max(0.0, 1.0 - std::abs(altitude - 25000.0) / 15000.0))};
}

// Spherical geometry around the planet's centre: r is the distance to it, mu the cosine between the
// ray and the local vertical.
double DistanceToTop(const Atmosphere& a, double r, double mu)
{
    const double discriminant = r * r * (mu * mu - 1.0) + a.top * a.top;
    return std::max(0.0, -r * mu + std::sqrt(std::max(discriminant, 0.0)));
}

double DistanceToBottom(const Atmosphere& a, double r, double mu)
{
    const double discriminant = r * r * (mu * mu - 1.0) + a.bottom * a.bottom;
    return std::max(0.0, -r * mu - std::sqrt(std::max(discriminant, 0.0)));
}

bool IntersectsGround(const Atmosphere& a, double r, double mu)
{
    return mu < 0.0 && r * r * (mu * mu - 1.0) + a.bottom * a.bottom >= 0.0;
}

// --- Transmittance table (Bruneton's parameterisation) --------------------------------------------

constexpr int kTransmittanceWidth = 256; // mu
constexpr int kTransmittanceHeight = 64; // r

double UnitFromTexel(double u, int size)
{
    return (u - 0.5 / size) / (1.0 - 1.0 / size);
}

double TexelFromUnit(double x, int size)
{
    return 0.5 / size + x * (1.0 - 1.0 / size);
}

struct Tables
{
    const Atmosphere* atmosphere = nullptr;
    std::vector<Spectrum> transmittance;      // kTransmittanceWidth * kTransmittanceHeight
    std::vector<Spectrum> multipleScattering; // kMultipleWidth * kMultipleHeight
};

void TransmittanceRMu(const Atmosphere& a, double u, double v, double& r, double& mu)
{
    const double h = std::sqrt(a.top * a.top - a.bottom * a.bottom);
    const double rho = h * UnitFromTexel(v, kTransmittanceHeight);
    r = std::sqrt(rho * rho + a.bottom * a.bottom);
    const double dMin = a.top - r;
    const double dMax = rho + h;
    const double d = dMin + UnitFromTexel(u, kTransmittanceWidth) * (dMax - dMin);
    mu = d <= 0.0 ? 1.0 : std::clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
}

Spectrum OpticalDepthToTop(const Atmosphere& a, double r, double mu)
{
    constexpr int kSteps = 500;
    const double dt = DistanceToTop(a, r, mu) / kSteps;
    std::array<double, kBins> depth{};
    for (int i = 0; i < kSteps; ++i)
    {
        const double t = (double(i) + 0.5) * dt;
        const double ri = std::sqrt(r * r + t * t + 2.0 * r * mu * t);
        const Densities d = DensitiesAt(ri - a.bottom);
        for (int b = 0; b < kBins; ++b)
        {
            depth[b] += (a.rayleigh[b] * d.rayleigh + a.mieExtinction[b] * d.mie + a.ozone[b] * d.ozone) * dt;
        }
    }
    Spectrum out{};
    for (int b = 0; b < kBins; ++b)
    {
        out[b] = float(depth[b]);
    }
    return out;
}

// Bilinear lookup of a spectral table between texel centres.
Spectrum Bilinear(const std::vector<Spectrum>& table, int width, int height, double u, double v)
{
    const double x = std::clamp(u * width - 0.5, 0.0, double(width - 1));
    const double y = std::clamp(v * height - 0.5, 0.0, double(height - 1));
    const int x0 = std::min(int(x), width - 2);
    const int y0 = std::min(int(y), height - 2);
    const float fx = float(x - x0);
    const float fy = float(y - y0);
    const Spectrum& a = table[size_t(y0) * width + x0];
    const Spectrum& b = table[size_t(y0) * width + x0 + 1];
    const Spectrum& c = table[size_t(y0 + 1) * width + x0];
    const Spectrum& d = table[size_t(y0 + 1) * width + x0 + 1];
    Spectrum out;
    for (int i = 0; i < kBins; ++i)
    {
        const float bottom = a[i] + (b[i] - a[i]) * fx;
        const float top = c[i] + (d[i] - c[i]) * fx;
        out[i] = bottom + (top - bottom) * fy;
    }
    return out;
}

Spectrum Transmittance(const Tables& tables, double r, double mu)
{
    const Atmosphere& a = *tables.atmosphere;
    const double h = std::sqrt(a.top * a.top - a.bottom * a.bottom);
    const double rho = std::sqrt(std::max(r * r - a.bottom * a.bottom, 0.0));
    const double d = DistanceToTop(a, r, mu);
    const double dMin = a.top - r;
    const double dMax = rho + h;
    const double xMu = dMax > dMin ? (d - dMin) / (dMax - dMin) : 0.0;
    return Bilinear(tables.transmittance, kTransmittanceWidth, kTransmittanceHeight,
                    TexelFromUnit(std::clamp(xMu, 0.0, 1.0), kTransmittanceWidth),
                    TexelFromUnit(std::clamp(rho / h, 0.0, 1.0), kTransmittanceHeight));
}

// Transmittance from a point towards the sun, zero when the planet is in the way.
Spectrum TransmittanceToSun(const Tables& tables, double r, double muSun)
{
    if (IntersectsGround(*tables.atmosphere, r, muSun))
    {
        return Spectrum{};
    }
    return Transmittance(tables, r, muSun);
}

// --- Multiple scattering table (Hillaire 2020, section 5.5) --------------------------------------

constexpr int kMultipleWidth = 32;  // sun zenith cosine, -1 .. 1
constexpr int kMultipleHeight = 32; // altitude, 0 .. top

Spectrum MultipleScattering(const Tables& tables, double r, double muSun)
{
    const Atmosphere& a = *tables.atmosphere;
    const double u = 0.5 + 0.5 * std::clamp(muSun, -1.0, 1.0);
    const double v = std::clamp((r - a.bottom) / (a.top - a.bottom), 0.0, 1.0);
    return Bilinear(tables.multipleScattering, kMultipleWidth, kMultipleHeight, u, v);
}

// Light scattered once (unit solar irradiance, isotropic phase) along all directions around a point
// of the atmosphere, and the fraction of light scattered there that it would get back: Psi_ms of
// Hillaire 2020, eq. 10 to 12 (the infinite series of orders as 1 / (1 - f_ms)).
Spectrum ComputeMultipleScattering(const Tables& tables, double r, double muSun)
{
    const Atmosphere& a = *tables.atmosphere;
    constexpr int kDirections = 8; // per axis: 64 directions over the sphere
    constexpr int kSteps = 40;
    const double sunX = std::sqrt(std::max(0.0, 1.0 - muSun * muSun));
    std::array<double, kBins> l2{};
    std::array<double, kBins> fms{};
    for (int i = 0; i < kDirections * kDirections; ++i)
    {
        const double cosTheta = 1.0 - 2.0 * ((double(i / kDirections) + 0.5) / kDirections);
        const double phi = 2.0 * kPi * ((double(i % kDirections) + 0.5) / kDirections);
        const double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
        // Point at (0, r, 0), sun in the xy plane, direction w.
        const double wx = sinTheta * std::cos(phi);
        const double wy = cosTheta;
        const double wz = sinTheta * std::sin(phi);
        const bool ground = IntersectsGround(a, r, wy);
        const double tMax = ground ? DistanceToBottom(a, r, wy) : DistanceToTop(a, r, wy);
        const double dt = tMax / kSteps;
        std::array<double, kBins> throughput;
        throughput.fill(1.0);
        std::array<double, kBins> l{};
        std::array<double, kBins> f{};
        for (int s = 0; s < kSteps; ++s)
        {
            const double t = (double(s) + 0.5) * dt;
            const double px = wx * t;
            const double py = r + wy * t;
            const double pz = wz * t;
            const double rp = std::sqrt(px * px + py * py + pz * pz);
            const double muSunP = (px * sunX + py * muSun) / rp;
            const Densities d = DensitiesAt(rp - a.bottom);
            const Spectrum sun = TransmittanceToSun(tables, rp, muSunP);
            for (int b = 0; b < kBins; ++b)
            {
                const double scattering = a.rayleigh[b] * d.rayleigh + a.mieScattering[b] * d.mie;
                const double extinction =
                    a.rayleigh[b] * d.rayleigh + a.mieExtinction[b] * d.mie + a.ozone[b] * d.ozone;
                const double step = std::exp(-extinction * dt);
                const double integral = extinction > 0.0 ? (1.0 - step) / extinction : dt;
                l[b] += throughput[b] * scattering * sun[b] / (4.0 * kPi) * integral;
                f[b] += throughput[b] * scattering * integral;
                throughput[b] *= step;
            }
        }
        if (ground)
        {
            const double gx = wx * tMax;
            const double gy = r + wy * tMax;
            const double gz = wz * tMax;
            const double rg = std::sqrt(gx * gx + gy * gy + gz * gz);
            const double muSunG = (gx * sunX + gy * muSun) / rg;
            const Spectrum sun = TransmittanceToSun(tables, a.bottom, muSunG);
            for (int b = 0; b < kBins; ++b)
            {
                l[b] += throughput[b] * sun[b] * std::max(muSunG, 0.0) * a.groundAlbedo / kPi;
            }
        }
        for (int b = 0; b < kBins; ++b)
        {
            l2[b] += l[b] / (kDirections * kDirections);
            fms[b] += f[b] / (kDirections * kDirections);
        }
    }
    Spectrum out{};
    for (int b = 0; b < kBins; ++b)
    {
        out[b] = float(l2[b] / std::max(1.0 - fms[b], 1e-6));
    }
    return out;
}

// --- Sky radiance --------------------------------------------------------------------------------

double RayleighPhase(double nu)
{
    return 3.0 / (16.0 * kPi) * (1.0 + nu * nu);
}

double CornetteShanksPhase(double nu, double g)
{
    const double g2 = g * g;
    return 3.0 / (8.0 * kPi) * (1.0 - g2) * (1.0 + nu * nu) /
           ((2.0 + g2) * std::pow(std::max(1.0 + g2 - 2.0 * g * nu, 1e-9), 1.5));
}

// A ray from the viewer and what it has gathered so far, per unit solar irradiance: the light the air
// scattered towards the viewer and the transmittance back to it.
struct ViewRay
{
    double dx = 0.0, dy = 0.0, dz = 0.0;       // direction
    double sunX = 0.0, sunY = 0.0, sunZ = 0.0; // towards the sun
    double phaseR = 0.0;                       // Rayleigh and Mie phase functions between the two
    double phaseM = 0.0;
    std::array<double, kBins> throughput{};
    std::array<double, kBins> l{};
};

ViewRay MakeViewRay(double dx, double dy, double dz, double sunX, double sunY, double sunZ)
{
    ViewRay ray{dx, dy, dz, sunX, sunY, sunZ};
    const double nu = dx * sunX + dy * sunY + dz * sunZ;
    ray.phaseR = RayleighPhase(nu);
    ray.phaseM = CornetteShanksPhase(nu, kMieG);
    ray.throughput.fill(1.0);
    return ray;
}

// One step of the ray march, the segment of length dt around distance t: single scattering of the
// sun and the multiple-scattering term of the point, then the segment's extinction.
void MarchStep(const Tables& tables, ViewRay& ray, double t, double dt)
{
    const Atmosphere& a = *tables.atmosphere;
    const double px = ray.dx * t;
    const double py = a.viewRadius + ray.dy * t;
    const double pz = ray.dz * t;
    const double rp = std::sqrt(px * px + py * py + pz * pz);
    const double muSunP = (px * ray.sunX + py * ray.sunY + pz * ray.sunZ) / rp;
    const Densities d = DensitiesAt(rp - a.bottom);
    const Spectrum sun = TransmittanceToSun(tables, rp, muSunP);
    const Spectrum multiple = MultipleScattering(tables, rp, muSunP);
    for (int b = 0; b < kBins; ++b)
    {
        const double rayleigh = a.rayleigh[b] * d.rayleigh;
        const double mie = a.mieScattering[b] * d.mie;
        const double extinction = rayleigh + a.mieExtinction[b] * d.mie + a.ozone[b] * d.ozone;
        const double source = (rayleigh * ray.phaseR + mie * ray.phaseM) * sun[b] + (rayleigh + mie) * multiple[b];
        const double step = std::exp(-extinction * dt);
        ray.l[b] += ray.throughput[b] * source * (extinction > 0.0 ? (1.0 - step) / extinction : dt);
        ray.throughput[b] *= step;
    }
}

// Where a ray from the viewer leaves the air: the ground or the top of the atmosphere.
double RayEnd(const Atmosphere& a, double dy, bool& ground)
{
    ground = IntersectsGround(a, a.viewRadius, dy);
    return ground ? DistanceToBottom(a, a.viewRadius, dy) : DistanceToTop(a, a.viewRadius, dy);
}

// Spectral radiance of the sky (no sun disc) along a direction from the viewer, per unit solar
// irradiance: single scattering of the sun, the multiple-scattering term of each point, and the
// planet's ground lit by the sun below the horizon.
Spectrum SkySpectrum(const Tables& tables, double dx, double dy, double dz, double sunX, double sunY, double sunZ)
{
    const Atmosphere& a = *tables.atmosphere;
    constexpr int kSteps = 96;
    bool ground = false;
    const double tMax = RayEnd(a, dy, ground);
    ViewRay ray = MakeViewRay(dx, dy, dz, sunX, sunY, sunZ);
    for (int s = 0; s < kSteps; ++s)
    {
        // Quadratic spacing: short steps near the viewer, where the air is densest.
        const double t0 = tMax * (double(s) / kSteps) * (double(s) / kSteps);
        const double t1 = tMax * (double(s + 1) / kSteps) * (double(s + 1) / kSteps);
        MarchStep(tables, ray, 0.5 * (t0 + t1), t1 - t0);
    }
    const std::array<double, kBins>& throughput = ray.throughput;
    std::array<double, kBins>& l = ray.l;
    if (ground)
    {
        const double gx = dx * tMax;
        const double gy = a.viewRadius + dy * tMax;
        const double gz = dz * tMax;
        const double rg = std::sqrt(gx * gx + gy * gy + gz * gz);
        const double muSunG = (gx * sunX + gy * sunY + gz * sunZ) / rg;
        const Spectrum sun = TransmittanceToSun(tables, a.bottom, muSunG);
        for (int b = 0; b < kBins; ++b)
        {
            l[b] += throughput[b] * sun[b] * std::max(muSunG, 0.0) * a.groundAlbedo / kPi;
        }
    }
    Spectrum out{};
    for (int b = 0; b < kBins; ++b)
    {
        out[b] = float(l[b]);
    }
    return out;
}

// W m^-2 sr^-1 nm^-1 per bin -> linear sRGB in nits (luminous efficacy 683 lm/W).
float3 ToRgb(const Atmosphere& a, const Spectrum& radiance)
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    for (int b = 0; b < kBins; ++b)
    {
        x += 683.0 * radiance[b] * a.cmf[b][0];
        y += 683.0 * radiance[b] * a.cmf[b][1];
        z += 683.0 * radiance[b] * a.cmf[b][2];
    }
    const double red = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    const double green = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    const double blue = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    return float3(float(std::max(red, 0.0)), float(std::max(green, 0.0)), float(std::max(blue, 0.0)));
}

void BuildTables(Tables& tables)
{
    const Atmosphere& a = *tables.atmosphere;
    tables.transmittance.resize(size_t(kTransmittanceWidth) * kTransmittanceHeight);
    ParallelFor(kTransmittanceHeight,
                [&](int y)
                {
                    for (int x = 0; x < kTransmittanceWidth; ++x)
                    {
                        double r = 0.0;
                        double mu = 0.0;
                        TransmittanceRMu(a, (x + 0.5) / kTransmittanceWidth, (y + 0.5) / kTransmittanceHeight, r, mu);
                        Spectrum depth = OpticalDepthToTop(a, r, mu);
                        for (float& value : depth)
                        {
                            value = std::exp(-value);
                        }
                        tables.transmittance[size_t(y) * kTransmittanceWidth + x] = depth;
                    }
                });
    tables.multipleScattering.resize(size_t(kMultipleWidth) * kMultipleHeight);
    ParallelFor(kMultipleHeight,
                [&](int y)
                {
                    const double r = a.bottom + (a.top - a.bottom) * (y + 0.5) / kMultipleHeight;
                    for (int x = 0; x < kMultipleWidth; ++x)
                    {
                        const double muSun = -1.0 + 2.0 * (x + 0.5) / kMultipleWidth;
                        tables.multipleScattering[size_t(y) * kMultipleWidth + x] =
                            ComputeMultipleScattering(tables, r, muSun);
                    }
                });
}

constexpr u32 kSkyWidth = 256;
constexpr u32 kSkyHeight = 256;

// Aerial perspective tables: directions as the sky table at a lower resolution, distances in
// quadratically spaced slices up to kAerialDistance (the first beyond the viewer at 104 m).
constexpr u32 kAerialWidth = 32;
constexpr u32 kAerialHeight = 64;
constexpr u32 kAerialSlices = 32;
constexpr double kAerialDistance = 100000.0;

// The light the air scatters towards the viewer along a direction (per unit solar irradiance) and
// its transmittance, from the viewer to each slice's distance; constant beyond the ground or the
// top of the atmosphere.
void AerialSpectra(const Tables& tables, double dx, double dy, double dz, double sunX, double sunY, double sunZ,
                   std::array<std::array<double, kBins>, kAerialSlices>& inscatter,
                   std::array<std::array<double, kBins>, kAerialSlices>& transmittance)
{
    constexpr int kSubsteps = 8;
    bool ground = false;
    const double end = RayEnd(*tables.atmosphere, dy, ground);
    ViewRay ray = MakeViewRay(dx, dy, dz, sunX, sunY, sunZ);
    inscatter[0] = ray.l;
    transmittance[0] = ray.throughput;
    for (u32 k = 1; k < kAerialSlices; ++k)
    {
        const double a = double(k - 1) / double(kAerialSlices - 1);
        const double b = double(k) / double(kAerialSlices - 1);
        const double t0 = std::min(kAerialDistance * a * a, end);
        const double t1 = std::min(kAerialDistance * b * b, end);
        const double dt = (t1 - t0) / kSubsteps;
        for (int s = 0; dt > 0.0 && s < kSubsteps; ++s)
        {
            MarchStep(tables, ray, t0 + (double(s) + 0.5) * dt, dt);
        }
        inscatter[k] = ray.l;
        transmittance[k] = ray.throughput;
    }
}

} // namespace

float3 SunDirection(const Sky& sky)
{
    const double elevation = double(sky.sunElevation) * kPi / 180.0;
    const double azimuth = double(sky.sunAzimuth) * kPi / 180.0;
    return float3(float(std::cos(elevation) * std::sin(azimuth)), float(std::sin(elevation)),
                  float(std::cos(elevation) * std::cos(azimuth)));
}

SkyTable BuildSkyTable(const Sky& sky, float3 uniform)
{
    SkyTable table;
    table.data.sunDirection = SunDirection(sky);
    table.data.sunCosRadius = 1.0f;
    table.data.sunRadiance = float3(0.0f);
    table.data.width = 0;
    table.data.height = 0;
    table.data.uniformRadiance = uniform;
    if (!sky.physical)
    {
        return table;
    }

    const Atmosphere atmosphere = MakeAtmosphere(sky);
    Tables tables;
    tables.atmosphere = &atmosphere;
    BuildTables(tables);

    // The sky as seen from the scene, in the frame where the sun has azimuth 0.
    const double sunElevation = double(sky.sunElevation) * kPi / 180.0;
    const double sunY = std::sin(sunElevation);
    const double sunZ = std::cos(sunElevation);
    table.radiance.resize(size_t(kSkyWidth) * kSkyHeight * 3);
    ParallelFor(int(kSkyHeight),
                [&](int y)
                {
                    const double v = (y + 0.5) / kSkyHeight;
                    const double w = 2.0 * v - 1.0;
                    const double elevation = (w < 0.0 ? -1.0 : 1.0) * 0.5 * kPi * w * w;
                    for (u32 x = 0; x < kSkyWidth; ++x)
                    {
                        const double u = (x + 0.5) / kSkyWidth;
                        const double phi = kPi * u * u;
                        const double dx = std::cos(elevation) * std::sin(phi);
                        const double dy = std::sin(elevation);
                        const double dz = std::cos(elevation) * std::cos(phi);
                        Spectrum radiance = SkySpectrum(tables, dx, dy, dz, 0.0, sunY, sunZ);
                        for (int b = 0; b < kBins; ++b)
                        {
                            radiance[b] *= atmosphere.solar[b];
                        }
                        const float3 rgb = ToRgb(atmosphere, radiance);
                        const size_t texel = (size_t(y) * kSkyWidth + x) * 3;
                        table.radiance[texel] = rgb.x;
                        table.radiance[texel + 1] = rgb.y;
                        table.radiance[texel + 2] = rgb.z;
                    }
                });
    table.data.width = kSkyWidth;
    table.data.height = kSkyHeight;

    // Aerial perspective. Its transmittance turns into RGB as the factor by which it changes the
    // colour of sunlight, channel by channel.
    const float3 white = ToRgb(atmosphere, atmosphere.solar);
    table.aerialWidth = kAerialWidth;
    table.aerialHeight = kAerialHeight;
    table.aerialInscatter.assign(size_t(kAerialWidth) * kAerialHeight * kAerialSlices * 4, 0);
    table.aerialTransmittance.assign(table.aerialInscatter.size(), 0);
    ParallelFor(int(kAerialHeight),
                [&](int y)
                {
                    const double v = (y + 0.5) / kAerialHeight;
                    const double w = 2.0 * v - 1.0;
                    const double elevation = (w < 0.0 ? -1.0 : 1.0) * 0.5 * kPi * w * w;
                    std::array<std::array<double, kBins>, kAerialSlices> inscatter;
                    std::array<std::array<double, kBins>, kAerialSlices> transmittance;
                    for (u32 x = 0; x < kAerialWidth; ++x)
                    {
                        const double u = (x + 0.5) / kAerialWidth;
                        const double phi = kPi * u * u;
                        AerialSpectra(tables, std::cos(elevation) * std::sin(phi), std::sin(elevation),
                                      std::cos(elevation) * std::cos(phi), 0.0, sunY, sunZ, inscatter, transmittance);
                        for (u32 k = 0; k < kAerialSlices; ++k)
                        {
                            Spectrum light{};
                            Spectrum through{};
                            for (int b = 0; b < kBins; ++b)
                            {
                                light[b] = float(inscatter[k][b] * atmosphere.solar[b]);
                                through[b] = float(transmittance[k][b] * atmosphere.solar[b]);
                            }
                            const float3 rgb = ToRgb(atmosphere, light);
                            const float3 t = ToRgb(atmosphere, through) / white;
                            u16* in = &table.aerialInscatter[((size_t(k) * kAerialHeight + y) * kAerialWidth + x) * 4];
                            u16* tr =
                                &table.aerialTransmittance[((size_t(k) * kAerialHeight + y) * kAerialWidth + x) * 4];
                            in[0] = FloatToHalf(rgb.x);
                            in[1] = FloatToHalf(rgb.y);
                            in[2] = FloatToHalf(rgb.z);
                            in[3] = FloatToHalf(0.0f);
                            tr[0] = FloatToHalf(std::clamp(t.x, 0.0f, 1.0f));
                            tr[1] = FloatToHalf(std::clamp(t.y, 0.0f, 1.0f));
                            tr[2] = FloatToHalf(std::clamp(t.z, 0.0f, 1.0f));
                            tr[3] = FloatToHalf(1.0f);
                        }
                    }
                });
    table.data.aerialDistance = float(kAerialDistance);
    table.data.aerialSlices = kAerialSlices;

    // The sun's disc: its radiance outside the atmosphere times the transmittance towards it.
    const double radius = double(sky.sunAngularRadius) * kPi / 180.0;
    const double solidAngle = 2.0 * kPi * (1.0 - std::cos(radius));
    const Spectrum transmittance = TransmittanceToSun(tables, atmosphere.viewRadius, sunY);
    Spectrum sun{};
    for (int b = 0; b < kBins; ++b)
    {
        sun[b] = float(atmosphere.solar[b] / solidAngle * transmittance[b]);
    }
    table.data.sunCosRadius = float(std::cos(radius));
    table.data.sunRadiance = ToRgb(atmosphere, sun);
    return table;
}

SkyModelProbe ProbeSkyModel(const Sky& sky)
{
    const Atmosphere a = MakeAtmosphere(sky);
    Tables tables;
    tables.atmosphere = &a;
    BuildTables(tables);
    // The bin holding 550 nm; the ozone layer integrates to 15 km of peak density.
    const int bin = int((550.0 - kFirstNm) / kBinNm);
    const double mieColumn = a.mieExtinction[bin] * kMieHeight * (1.0 - std::exp(-(a.top - a.bottom) / kMieHeight));
    const double rayleighColumn =
        a.rayleigh[bin] * kRayleighHeight * (1.0 - std::exp(-(a.top - a.bottom) / kRayleighHeight));
    SkyModelProbe probe;
    probe.zenithOpticalDepth550 = float(rayleighColumn + mieColumn + a.ozone[bin] * 15000.0);
    probe.zenithTransmittance550 = Transmittance(tables, a.bottom, 1.0)[bin];
    double illuminance = 0.0;
    for (int b = 0; b < kBins; ++b)
    {
        illuminance += 683.0 * a.solar[b] * a.cmf[b][1];
    }
    probe.sunIlluminanceOutside = float(illuminance);
    return probe;
}

} // namespace air::renderer
