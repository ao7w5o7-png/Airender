#include "air/core/Image.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <string>

namespace air {

namespace {

float LinearToSrgb(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    return v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

float SrgbToLinear(float v) {
    return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

// Reads the next whitespace separated token of a PNM/PFM header, skipping '#' comments.
bool ReadToken(std::istream& in, std::string& token) {
    token.clear();
    char c;
    while (in.get(c)) {
        if (c == '#') {
            std::string ignored;
            std::getline(in, ignored);
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            token += c;
            break;
        }
    }
    while (in.get(c) && !std::isspace(static_cast<unsigned char>(c))) {
        token += c;
    }
    return !token.empty();
}

} // namespace

bool WritePfm(const std::filesystem::path& path, const Image& image) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    const bool littleEndian = std::endian::native == std::endian::little;
    out << "PF\n" << image.width << ' ' << image.height << '\n' << (littleEndian ? "-1.0" : "1.0") << '\n';
    // PFM stores rows bottom to top.
    std::vector<float> row(size_t(image.width) * 3);
    for (u32 y = image.height; y-- > 0;) {
        for (u32 x = 0; x < image.width; ++x) {
            const float* p = image.At(x, y);
            row[x * 3 + 0] = p[0];
            row[x * 3 + 1] = p[1];
            row[x * 3 + 2] = p[2];
        }
        out.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size() * sizeof(float)));
    }
    return bool(out);
}

std::optional<Image> ReadPfm(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::string magic, w, h, scale;
    if (!in || !ReadToken(in, magic) || (magic != "PF" && magic != "Pf") || !ReadToken(in, w) || !ReadToken(in, h) ||
        !ReadToken(in, scale)) {
        return std::nullopt;
    }
    const u32 channels = magic == "PF" ? 3 : 1;
    Image image(u32(std::stoul(w)), u32(std::stoul(h)));
    const bool fileLittle = std::stof(scale) < 0.0f;
    const bool swap = fileLittle != (std::endian::native == std::endian::little);
    std::vector<float> row(size_t(image.width) * channels);
    for (u32 y = image.height; y-- > 0;) {
        if (!in.read(reinterpret_cast<char*>(row.data()), std::streamsize(row.size() * sizeof(float)))) {
            return std::nullopt;
        }
        for (float& f : row) {
            if (swap) {
                u32 bits;
                std::memcpy(&bits, &f, 4);
                bits = (bits >> 24) | ((bits >> 8) & 0xFF00) | ((bits << 8) & 0xFF0000) | (bits << 24);
                std::memcpy(&f, &bits, 4);
            }
        }
        for (u32 x = 0; x < image.width; ++x) {
            float* p = image.At(x, y);
            for (u32 c = 0; c < 3; ++c) {
                p[c] = row[x * channels + (channels == 3 ? c : 0)];
            }
            p[3] = 1.0f;
        }
    }
    return image;
}

bool WritePpm(const std::filesystem::path& path, const Image& image) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    std::vector<u8> row(size_t(image.width) * 3);
    for (u32 y = 0; y < image.height; ++y) {
        for (u32 x = 0; x < image.width; ++x) {
            const float* p = image.At(x, y);
            for (u32 c = 0; c < 3; ++c) {
                row[x * 3 + c] = u8(std::lround(LinearToSrgb(p[c]) * 255.0f));
            }
        }
        out.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size()));
    }
    return bool(out);
}

std::optional<Image> ReadPpm(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::string magic, w, h, maxValue;
    if (!in || !ReadToken(in, magic) || magic != "P6" || !ReadToken(in, w) || !ReadToken(in, h) ||
        !ReadToken(in, maxValue) || std::stoi(maxValue) != 255) {
        return std::nullopt;
    }
    Image image(u32(std::stoul(w)), u32(std::stoul(h)));
    std::vector<u8> row(size_t(image.width) * 3);
    for (u32 y = 0; y < image.height; ++y) {
        if (!in.read(reinterpret_cast<char*>(row.data()), std::streamsize(row.size()))) {
            return std::nullopt;
        }
        for (u32 x = 0; x < image.width; ++x) {
            float* p = image.At(x, y);
            for (u32 c = 0; c < 3; ++c) {
                p[c] = SrgbToLinear(row[x * 3 + c] / 255.0f);
            }
            p[3] = 1.0f;
        }
    }
    return image;
}

std::optional<Image> ReadImage(const std::filesystem::path& path) {
    const std::string ext = path.extension().string();
    if (ext == ".pfm") {
        return ReadPfm(path);
    }
    if (ext == ".ppm") {
        return ReadPpm(path);
    }
    return std::nullopt;
}

ImageCompareResult CompareImages(const Image& reference, const Image& test, const ImageCompareOptions& options) {
    ImageCompareResult result;
    if (reference.width != test.width || reference.height != test.height || reference.Empty()) {
        result.sizeMismatch = true;
        result.passed = false;
        return result;
    }
    result.diff = Image(reference.width, reference.height);
    double sumSq = 0.0;
    for (u32 y = 0; y < reference.height; ++y) {
        for (u32 x = 0; x < reference.width; ++x) {
            const float* a = reference.At(x, y);
            const float* b = test.At(x, y);
            float pixelMax = 0.0f;
            for (u32 c = 0; c < 3; ++c) {
                const float d = std::fabs(a[c] - b[c]);
                sumSq += double(d) * d;
                pixelMax = std::max(pixelMax, d);
            }
            result.maxAbsDiff = std::max(result.maxAbsDiff, double(pixelMax));
            if (pixelMax > options.pixelThreshold) {
                ++result.differingPixels;
            }
            float* out = result.diff.At(x, y);
            out[0] = std::min(1.0f, pixelMax * 10.0f);
            out[3] = 1.0f;
        }
    }
    const double pixelCount = double(reference.width) * reference.height;
    result.rmse = std::sqrt(sumSq / (pixelCount * 3.0));
    result.psnr = result.rmse > 0.0 ? 20.0 * std::log10(1.0 / result.rmse) : std::numeric_limits<double>::infinity();
    result.differingFraction = double(result.differingPixels) / pixelCount;
    result.passed = result.rmse <= options.maxRmse && result.differingFraction <= options.maxDifferingFraction;
    return result;
}

std::string ImageCompareResult::Summary() const {
    if (sizeMismatch) {
        return "FAIL: image sizes differ";
    }
    return std::format("{}: rmse={:.6f} psnr={:.2f}dB maxAbsDiff={:.6f} differingPixels={} ({:.4f}%)",
                       passed ? "PASS" : "FAIL", rmse, psnr, maxAbsDiff, differingPixels, differingFraction * 100.0);
}

} // namespace air
